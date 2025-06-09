#pragma once
#include <string>

/**
 * @brief libcurl write callback to append data into a std::string.
 * @param contents Pointer to received data.
 * @param size      Size of each data element.
 * @param nmemb     Number of elements.
 * @param data      Destination string.
 * @return Number of bytes processed.
 */
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* data);

class UpdateManager {
public:
    /**
     * @brief Checks GitHub for a newer release tag.
     * @return True if a newer version is available.
     */
    static bool CheckForUpdate();

    /**
     * @brief Downloads the latest release archive to a temporary path.
     * @return True on successful download and checksum verification.
     */
    static bool DownloadUpdate();

    /**
     * @brief Executes the update installation script and exits.
     *        This will replace the running binary with the new version.
     */
    static void ApplyUpdate();

    /**
     * @brief Reads the current version from the bundled version.info file.
     * @return The current version string, or "0.0.0" if not found.
     */
    static std::string GetCurrentVersion();

    /**
     * @brief Fetches the latest GitHub release tag name.
     * @return Tag string (e.g. "v1.2.3") or empty on failure.
     */
    static std::string GetLatestVersion();
    
private:
    static std::string current_version;
    static std::string latest_version;

    /**
     * @brief Verifies the downloaded file's checksum.
     *        Currently is a mock-function.
     * @param path Local path to the downloaded archive.
     * @return True if checksum matches (stubbed to always true).
     */
    static bool VerifyChecksum(const std::string& path);
};