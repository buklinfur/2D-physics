#pragma once
#include <string>

#ifndef LBMSIM_UPDATE_REPO
/// GitHub "owner/name" that releases are fetched from. Override with -DLBMSIM_UPDATE_REPO=... in CMake.
#define LBMSIM_UPDATE_REPO "buklinfur/2D-physics"
#endif

/**
 * @brief Self-update through GitHub Releases.
 *
 * Flow: CheckForUpdate() reads the latest release from the GitHub REST API,
 * DownloadUpdate() fetches the release asset `lbm-linux-latest.tar.gz` and
 * verifies its SHA-256 against the digest GitHub publishes for that asset,
 * ApplyUpdate() unpacks it next to the running binary with
 * UpdateLoader/scripts/update.sh and re-executes the new binary.
 */
class UpdateManager {
public:
    /** @brief Name of the release asset that contains the Linux build. */
    static constexpr const char* kAssetName = "lbm-linux-latest.tar.gz";

    /** @brief Local path the archive is downloaded to. */
    static constexpr const char* kDownloadPath = "/tmp/lbm_update.tar.gz";

    /**
     * @brief Checks GitHub for a release whose tag differs from the bundled version.
     * @return True if a different (newer) release tag is available.
     */
    static bool CheckForUpdate();

    /**
     * @brief Downloads the release asset and verifies its SHA-256 digest.
     * @return True only if the download succeeded and the checksum matches.
     */
    static bool DownloadUpdate();

    /**
     * @brief Installs the downloaded archive and re-executes the new binary.
     *
     * Runs UpdateLoader/scripts/update.sh (located next to the executable),
     * which unpacks the archive and atomically replaces the binary and
     * version.info. On success the process image is replaced via execv() with
     * the same arguments; on failure the function returns and the current
     * version keeps running.
     *
     * @param argv Original argv, passed to the new binary.
     */
    static void ApplyUpdate(char** argv);

    /**
     * @brief Reads the current version from the version.info file next to the binary.
     * @return The current version string, or "0.0.0" if not found.
     */
    static std::string GetCurrentVersion();

    /**
     * @brief Fetches (once) and returns the latest GitHub release tag name.
     * @return Tag string (e.g. "v0.1.0-master") or empty on failure.
     */
    static std::string GetLatestVersion();

    /**
     * @brief Computes the SHA-256 of a file.
     * @param path File to hash.
     * @return Lower-case hex digest, or empty string if the file cannot be read.
     */
    static std::string Sha256File(const std::string& path);

private:
    static std::string current_version;
    static std::string latest_version;
    static std::string asset_url;     ///< browser_download_url of kAssetName in the latest release.
    static std::string asset_sha256;  ///< Expected digest of the asset (hex), from the release metadata.

    /**
     * @brief Downloads and parses the latest release metadata.
     * @return True if the tag name was obtained.
     */
    static bool FetchLatestRelease();

    /**
     * @brief Verifies the downloaded file against the published SHA-256 digest.
     *        Fails closed: a missing digest is treated as a mismatch.
     * @param path Local path to the downloaded archive.
     * @return True if the checksum matches.
     */
    static bool VerifyChecksum(const std::string& path);

    /** @brief Directory that contains the running executable. */
    static std::string ExecutableDir();
};
