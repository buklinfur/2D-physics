#include "UpdateLoader.hpp"
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <unistd.h>

std::string UpdateManager::current_version = "";
std::string UpdateManager::latest_version  = "";

static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* data) {
    data->append(reinterpret_cast<char*>(contents), size * nmemb);
    return size * nmemb;
}

std::string UpdateManager::GetCurrentVersion() {
    if (!current_version.empty()) 
        return current_version;

    char exePath[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", exePath, sizeof(exePath) - 1);
    if (len != -1) {
        exePath[len] = '\0';
        auto versionPath = std::filesystem::path(exePath).parent_path() / "version.info";
        std::ifstream verFile(versionPath);
        if (verFile.good()) {
            std::getline(verFile, current_version);
        }
    }

    if (current_version.empty()) {
        current_version = "0.0.0";
    }
    return current_version;
}

bool UpdateManager::CheckForUpdate() {
    current_version = GetCurrentVersion();
    latest_version  = GetLatestVersion();

    std::cout << "Current version: " << current_version << "\n"
              << "Latest version : " << latest_version << std::endl;

    return (!latest_version.empty() && current_version != latest_version);
}

bool UpdateManager::DownloadUpdate() {
    std::string url     = "https://github.com/buklinfur/2D-physics/releases/download/"
                          + latest_version + "/lbm-linux-latest.tar.gz";
    std::string tmpPath = "/tmp/lbm_update.tar.gz";

    CURL* curl = curl_easy_init();
    if (!curl) return false;

    FILE* fp = fopen(tmpPath.c_str(), "wb");
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, nullptr);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "LBM-UpdateAgent");
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

    CURLcode res = curl_easy_perform(curl);
    fclose(fp);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        std::cerr << "Download failed: " << curl_easy_strerror(res) << std::endl;
        return false;
    }
    return VerifyChecksum(tmpPath);
}

bool UpdateManager::VerifyChecksum(const std::string& path) {
    std::cout << "Skipping checksum verification for debugging" << std::endl;
    return true;
}

void UpdateManager::ApplyUpdate() {
    std::cout << "Applying update..." << std::endl;
    system("sh UpdateLoader/scripts/update.sh");
    exit(0);
}

std::string UpdateManager::GetLatestVersion() {
    CURL* curl       = curl_easy_init();
    std::string resp;

    if (curl) {
        curl_easy_setopt(curl, CURLOPT_URL, 
            "https://api.github.com/repos/buklinfur/2D-physics/releases/latest");
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &resp);
        curl_easy_setopt(curl, CURLOPT_USERAGENT, "LBM-UpdateAgent");
        curl_easy_perform(curl);
        curl_easy_cleanup(curl);
    }

    try {
        auto j = nlohmann::json::parse(resp);
        return j.at("tag_name").get<std::string>();
    } catch (...) {
        return "";
    }
}
