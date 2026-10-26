#include "ImageLoader.h"
#include <filesystem>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utility.hpp>

int LoadImages(const string& folder, vector<Mat>& images, vector<string>& names)
{
    vector<string> paths;
    glob(folder + "*", paths);

    for (const string& path : paths)
    {
        Mat img = imread(path, IMREAD_COLOR);
        if (img.empty())
        {
            cout << "skipping: " << path << endl;
            continue;
        }

        images.push_back(img);
        names.push_back(filesystem::path(path).filename().string());
    }

    cout << "'loaded " << images.size() << " images from " << folder << endl;
    return (int)images.size();

}
