#include "UpdateLoader.hpp"

#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <openssl/evp.h>

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unistd.h>

std::string UpdateManager::current_version;
std::string UpdateManager::latest_version;
std::string UpdateManager::asset_url;
std::string UpdateManager::asset_sha256;

namespace {

/// libcurl write callback that appends the received bytes to a std::string.
size_t WriteToString(void* contents, size_t size, size_t nmemb, void* userp) {
    static_cast<std::string*>(userp)->append(static_cast<char*>(contents), size * nmemb);
    return size * nmemb;
}

/// Shared libcurl options: identify ourselves, follow redirects (a renamed
/// repository answers with 301), fail on HTTP errors and do not hang forever.
void SetCommonOptions(CURL* curl, const std::string& url) {
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "LBM-UpdateAgent");
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L);
}

}  // namespace

std::string UpdateManager::ExecutableDir() {
    char exePath[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", exePath, sizeof(exePath) - 1);
    if (len == -1) {
        return ".";
    }
    exePath[len] = '\0';
    return std::filesystem::path(exePath).parent_path().string();
}

std::string UpdateManager::GetCurrentVersion() {
    if (!current_version.empty()) {
        return current_version;
    }

    std::ifstream verFile(std::filesystem::path(ExecutableDir()) / "version.info");
    if (verFile.good()) {
        std::getline(verFile, current_version);
    }
    if (current_version.empty()) {
        current_version = "0.0.0";
    }
    return current_version;
}

bool UpdateManager::FetchLatestRelease() {
    CURL* curl = curl_easy_init();
    if (!curl) {
        return false;
    }

    std::string resp;
    SetCommonOptions(curl, std::string("https://api.github.com/repos/") + LBMSIM_UPDATE_REPO +
                               "/releases/latest");
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 15L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteToString);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &resp);
    const CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);
    if (res != CURLE_OK) {
        std::cerr << "Update check failed: " << curl_easy_strerror(res) << std::endl;
        return false;
    }

    try {
        const auto j = nlohmann::json::parse(resp);
        latest_version = j.at("tag_name").get<std::string>();
        asset_url.clear();
        asset_sha256.clear();
        for (const auto& asset : j.value("assets", nlohmann::json::array())) {
            if (asset.value("name", "") != kAssetName) {
                continue;
            }
            asset_url = asset.value("browser_download_url", "");
            const std::string digest = asset.value("digest", "");  // "sha256:<hex>"
            const std::string prefix = "sha256:";
            if (digest.compare(0, prefix.size(), prefix) == 0) {
                asset_sha256 = digest.substr(prefix.size());
            }
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Update check: unexpected release metadata (" << e.what() << ")" << std::endl;
        latest_version.clear();
        return false;
    }
}

std::string UpdateManager::GetLatestVersion() {
    if (latest_version.empty()) {
        FetchLatestRelease();
    }
    return latest_version;
}

bool UpdateManager::CheckForUpdate() {
    current_version = GetCurrentVersion();
    latest_version = GetLatestVersion();
    return !latest_version.empty() && current_version != latest_version;
}

bool UpdateManager::DownloadUpdate() {
    if (latest_version.empty() && !FetchLatestRelease()) {
        return false;
    }
    if (asset_url.empty()) {
        std::cerr << "Release " << latest_version << " has no asset named " << kAssetName << std::endl;
        return false;
    }

    CURL* curl = curl_easy_init();
    if (!curl) {
        return false;
    }
    FILE* fp = std::fopen(kDownloadPath, "wb");
    if (!fp) {
        curl_easy_cleanup(curl);
        std::cerr << "Cannot write " << kDownloadPath << std::endl;
        return false;
    }

    SetCommonOptions(curl, asset_url);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);  // default write function fwrite()s to fp
    const CURLcode res = curl_easy_perform(curl);
    std::fclose(fp);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        std::cerr << "Download failed: " << curl_easy_strerror(res) << std::endl;
        std::remove(kDownloadPath);
        return false;
    }
    if (!VerifyChecksum(kDownloadPath)) {
        std::remove(kDownloadPath);
        return false;
    }
    return true;
}

std::string UpdateManager::Sha256File(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return "";
    }

    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (!ctx || EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) != 1) {
        EVP_MD_CTX_free(ctx);
        return "";
    }
    char buf[1 << 16];
    while (in.read(buf, sizeof(buf)) || in.gcount() > 0) {
        EVP_DigestUpdate(ctx, buf, static_cast<size_t>(in.gcount()));
    }
    unsigned char md[EVP_MAX_MD_SIZE];
    unsigned int md_len = 0;
    EVP_DigestFinal_ex(ctx, md, &md_len);
    EVP_MD_CTX_free(ctx);

    static const char* hex = "0123456789abcdef";
    std::string out;
    out.reserve(md_len * 2);
    for (unsigned int i = 0; i < md_len; ++i) {
        out.push_back(hex[md[i] >> 4]);
        out.push_back(hex[md[i] & 0xF]);
    }
    return out;
}

bool UpdateManager::VerifyChecksum(const std::string& path) {
    if (asset_sha256.empty()) {
        std::cerr << "Release metadata has no SHA-256 digest for " << kAssetName
                  << "; refusing to install an unverified update." << std::endl;
        return false;
    }
    const std::string actual = Sha256File(path);
    if (actual != asset_sha256) {
        std::cerr << "SHA-256 mismatch: expected " << asset_sha256 << ", got "
                  << (actual.empty() ? "<unreadable>" : actual) << std::endl;
        return false;
    }
    std::cout << "SHA-256 verified: " << actual << std::endl;
    return true;
}

void UpdateManager::ApplyUpdate(char** argv) {
    const std::string dir = ExecutableDir();
    const std::string script = dir + "/UpdateLoader/scripts/update.sh";
    if (!std::filesystem::exists(script)) {
        std::cerr << "Update script not found: " << script << std::endl;
        return;
    }

    std::cout << "Applying update..." << std::endl;
    const std::string cmd = "sh '" + script + "' '" + dir + "' '" + kDownloadPath + "'";
    if (std::system(cmd.c_str()) != 0) {
        std::cerr << "Update script failed; the current version was left in place." << std::endl;
        return;
    }

    // The script replaced the binary with rename(), so the running image is
    // untouched; start the new one in place of this process.
    const std::string exe = dir + "/executables";
    std::cout << "Restarting " << exe << std::endl;
    execv(exe.c_str(), argv);
    std::perror("execv");
}
