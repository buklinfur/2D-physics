#include "UpdateLoader.hpp"
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>
#include <iostream>

std::string UpdateManager::current_version = "";
std::string UpdateManager::latest_version = "";

static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* data) {
    data->append((char*)contents, size * nmemb);
    return size * nmemb;
}

std::string UpdateManager::GetCurrentVersion() {
    if (!current_version.empty()) return current_version;
    
    std::ifstream verFile("version.info");
    if (verFile.good()) {
        std::getline(verFile, current_version);
    } else {
        current_version = "0.0.0";
    }
    return current_version;
}

bool UpdateManager::CheckForUpdate() {
    current_version = GetCurrentVersion();
    latest_version = GetLatestVersion();
    
    std::cout << "Current version: " << current_version << std::endl;
    std::cout << "Latest version: " << latest_version << std::endl;
    
    // Compare semantic versions
    return (current_version != latest_version) && !latest_version.empty();
}

bool UpdateManager::DownloadUpdate() {
    std::string url = "https://github.com/yourusername/yourrepo/releases/download/" + 
                      latest_version + "/lbm-linux-Release-" + latest_version + ".tar.gz";
    
    std::string tmpPath = "/tmp/lbm_update.tar.gz";
    
    CURL* curl = curl_easy_init();
    if (!curl) return false;
    
    FILE* fp = fopen(tmpPath.c_str(), "wb");
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
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

// Add this new function
bool UpdateManager::VerifyChecksum(const std::string& path) {
    // In practice, you should compare with a checksum from GitHub API
    std::cout << "Skipping checksum verification for debugging" << std::endl;
    return true; // Temporarily skip for debugging
    
    // Actual implementation would:
    // 1. Fetch checksum from GitHub API
    // 2. Calculate local file checksum
    // 3. Compare them
}

void UpdateManager::ApplyUpdate() {
    std::cout << "Applying update..." << std::endl;
    system("sh UpdateLoader/scripts/update.sh");
    exit(0);
}

std::string UpdateManager::GetLatestVersion() {
    CURL* curl = curl_easy_init();
    std::string response;
    
    if(curl) {
        curl_easy_setopt(curl, CURLOPT_URL, "https://api.github.com/repos/buklinfur/2D-physics/releases/latest");
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
        curl_easy_setopt(curl, CURLOPT_USERAGENT, "LBM-UpdateAgent");
        
        curl_easy_perform(curl);
        curl_easy_cleanup(curl);
    }
    
    try {
        auto json = nlohmann::json::parse(response);
        return json["tag_name"];
    } catch(...) {
        return "";
    }
}