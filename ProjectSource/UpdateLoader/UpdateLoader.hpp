#pragma once

#include <string>

class UpdateLoader {
public:
  UpdateLoader(std::string repoOwner,
               std::string repoName,
               std::string currentVersion);

  std::string fetchLatestVersion();

  bool downloadReleaseAsset(const std::string& version,
                            const std::string& assetName,
                            const std::string& outPath);

  bool checkAndUpdate();

private:
  std::string owner_;
  std::string repo_;
  std::string currVer_;

  bool isVersionGreater(const std::string& a, const std::string& b);

  std::string httpGet(const std::string& url);
};
