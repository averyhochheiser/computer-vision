#pragma once
#include <string>
#include <vector>
#include <opencv2/core/mat.hpp>

#include "Utilities.h"

// reads image(s) and returns the number or images loaded
int LoadImages(const string& folder, vector<Mat>& images, vector<string>& names);
