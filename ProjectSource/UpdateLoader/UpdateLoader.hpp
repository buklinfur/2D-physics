#pragma once
#include <string>

class UpdateManager {
public:
    static bool CheckForUpdate();
    static bool DownloadUpdate();
    static void ApplyUpdate();
    static std::string GetCurrentVersion();
    static std::string GetLatestVersion();
    
private:
    static std::string current_version;
    static std::string latest_version;
    static bool VerifyChecksum(const std::string& path);
};