// UpdateLoader.cpp
#include "UpdateLoader.hpp"
#include <nlohmann/json.hpp>
// #include <curl/curl.h>
#include <iostream>
#include <fstream>

// UpdateLoader::UpdateLoader(std::string repoOwner,
//                            std::string repoName,
//                            std::string currentVersion)
//   : owner_(std::move(repoOwner)),
//     repo_(std::move(repoName)),
//     currVer_(std::move(currentVersion))
// {}

// std::string UpdateLoader::httpGet(const std::string& url) {
//   CURL* curl = curl_easy_init();
//   std::string data;
//   if (!curl) throw std::runtime_error("libcurl init failed");

//   curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
//   curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION,
//     +[](char* ptr, size_t size, size_t nmemb, void* userdata) {
//       auto& s = *static_cast<std::string*>(userdata);
//       s.append(ptr, size * nmemb);
//       return size * nmemb;
//     }
//   );
//   curl_easy_setopt(curl, CURLOPT_WRITEDATA, &data);

//   CURLcode res = curl_easy_perform(curl);
//   curl_easy_cleanup(curl);
//   if (res != CURLE_OK)
//     throw std::runtime_error("HTTP request failed");

//   return data;
// }

// std::string UpdateLoader::fetchLatestVersion() {
//   std::string apiUrl =
//     "https://api.github.com/repos/" + owner_ + "/" + repo_ + "/releases/latest";

//   std::string body = httpGet(apiUrl);
//   auto j = nlohmann::json::parse(body);
//   return j.at("tag_name").get<std::string>();  // e.g. "v1.2.4"
// }

// bool UpdateLoader::isVersionGreater(const std::string& a,
//                                     const std::string& b) {
//   // Simple lexicographic semver: strip leading 'v', split on '.', compare ints.
//   auto strip = [](const std::string& s){
//     return s.front()=='v' ? s.substr(1) : s;
//   };
//   std::vector<int> va, vb;
//   for (auto& x : {/*a*/}){} // implement split & stoi...
//   // [omitted for brevity]
//   return va > vb;
// }

// bool UpdateLoader::downloadReleaseAsset(const std::string& version,
//                                         const std::string& assetName,
//                                         const std::string& outPath)
// {
//   std::string url =
//     "https://github.com/" + owner_ + "/" + repo_ +
//     "/releases/download/" + version + "/" + assetName;

//   std::string fileData = httpGet(url);
//   std::ofstream out{outPath, std::ios::binary};
//   if (!out) return false;
//   out.write(fileData.data(), fileData.size());
//   return true;
// }

// bool UpdateLoader::checkAndUpdate() {
//   try {
//     std::string latest = fetchLatestVersion();
//     if (isVersionGreater(latest, currVer_)) {
//       std::cout << "New version " << latest << " available (you have "
//                 << currVer_ << "). Update? [Y/n] ";

//       char resp='y';
//       std::cin >> resp;
//       if (resp=='Y' || resp=='y') {
//         // decide your assetName & outPath based on OS/arch
//         std::string asset = /* e.g. "lbm-linux-Release-" + latest + ".tar.gz" */;
//         std::string out   = "update.tmp";
//         if (downloadReleaseAsset(latest, asset, out)) {
//           // unpack & replace binary; or instruct user
//           std::cout<<"Downloaded to "<<out<<". Please unpack and replace binary.\n";
//           return true;
//         }
//         else std::cerr<<"Download failed\n";
//       }
//     }
//   }
//   catch (const std::exception& e) {
//     std::cerr<<"Update check failed: "<<e.what()<<"\n";
//   }
//   return false;
// }
