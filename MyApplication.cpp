// Entry point for your application (menu option 'm')

#include "Utilities.h"
#include <filesystem>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utility.hpp>

const int TABLE_WIDTH = 1434;
const int TABLE_HEIGHT = 683;
const float BALL_RADIUS = 21.5f;

const float RADIUS_TOLERANCE = 0.15f;
const float OUTLINE_FRACTION = 0.25f;

const float HUE_WEIGHT = 255.0f / 179.0f;

const int WHITE_MAX_SATURATION = 80;
const int WHITE_MIN_VAL = 120;
const double WHITE_WEIGHT = 255;
const double RING_INNER = 0.5;
const double RING_OUTER = 0.95;

struct BallFeat
{
	double white;
	Scalar color;
};

float BallRadiusInPhoto(Point2f position_on_table, const Mat& perspective_matrix)
{
	vector<Point2f> table_points = {position_on_table, Point2f(position_on_table.x + BALL_RADIUS, position_on_table.y)};
	vector<Point2f> photo_points;
	perspectiveTransform(table_points, photo_points, perspective_matrix.inv());

	return (float)norm(photo_points[1] - photo_points[0]);
}

void BallRadiusRange(const Mat& perspective_matrix, int& min_rad, int& max_rad)
{
	Point2f corners[4] = {Point2f(0, 0), Point2f(TABLE_WIDTH, 0),
		Point2f(TABLE_WIDTH, TABLE_HEIGHT), Point2f(0, TABLE_HEIGHT)};

	float small = 1e9f, large = 0;
	for (int i = 0; i < 4; i++)
	{
		float radius = BallRadiusInPhoto(corners[i], perspective_matrix);
		small = min(small, radius);
		large = max(large, radius);
	}

	min_rad = (int)(small * (1 - RADIUS_TOLERANCE));
	max_rad = (int)ceil(large * (1 + RADIUS_TOLERANCE));
}

double Otsu(const Mat& image, const Mat& mask)
{
	vector<uchar> values;
	for (int row = 0; row < image.rows; row++)
		for (int col = 0; col < image.cols; col++)
			if (mask.at<uchar>(row, col))
				values.push_back(image.at<uchar>(row, col));

	Mat samples(values);
	Mat dummy;
	return threshold(samples, dummy, 0, 255, THRESH_BINARY | THRESH_OTSU);
}

Mat GetPerspectiveMatrix()
{
	Point2f source[4], destination[4];

	source[0] = Point2f(337, 35);
	source[1] = Point2f(1570, 21);
	source[2] = Point2f(1888, 451);
	source[3] = Point2f(38, 472);

	destination[0] = Point2f(0, 0);
	destination[1] = Point2f(TABLE_WIDTH - 1, 0);
	destination[2] = Point2f(TABLE_WIDTH - 1, TABLE_HEIGHT - 1);
	destination[3] = Point2f(0, TABLE_HEIGHT - 1);

	Mat perspective_matrix = getPerspectiveTransform(source, destination);
	return perspective_matrix;
}

Mat WarpToTable(const Mat& image, const Mat& perspective_matrix)
{
	Mat result = Mat(TABLE_HEIGHT, TABLE_WIDTH, image.type());
	warpPerspective(image, result, perspective_matrix, result.size());

	return result;
}

vector<Point2f> ToTableCoordinates(const vector<Point2f>& image_points, const Mat& perspective_matrix)
{
	vector<Point2f> table_points;
	if (!image_points.empty())
		perspectiveTransform(image_points, table_points, perspective_matrix);

	return table_points;
}

int HistogramPeak(const Mat& channel, const Mat& mask)
{
	int bins = 256;
	float range[] = {0, 256};
	const float *ranges[] = {range};
	int channel_index = 0;

	Mat histogram;
	calcHist(&channel, 1, &channel_index, mask, histogram, 1, &bins, ranges);

	Point peak;
	minMaxLoc(histogram, 0, 0, 0, &peak);

	return peak.y;
}

Mat TableMask(const Size& image_size, const Mat& perspective_matrix)
{
	Mat table = Mat(TABLE_HEIGHT, TABLE_WIDTH, CV_8UC1, Scalar(255));

	Mat mask;
	warpPerspective(table, mask, perspective_matrix, image_size, INTER_NEAREST | WARP_INVERSE_MAP);

	return mask;
}

Mat DifferenceFromCloth(const Mat& image, const Mat& perspective_matrix)
{
	Mat smooth;
	GaussianBlur(image, smooth, Size(5, 5), 0 );

	Mat hsv;
	cvtColor(smooth, hsv, COLOR_BGR2HSV);

	Mat table = TableMask(image.size(), perspective_matrix);

	vector<Mat> channels;
	split(hsv, channels);
	int hue = HistogramPeak(channels[0], table);
	int saturation = HistogramPeak(channels[1], table);
	int value = HistogramPeak(channels[2], table);

	Mat h, s, v;
	channels[0].convertTo(h, CV_32F);
	channels[1].convertTo(s, CV_32F);
	channels[2].convertTo(v, CV_32F);

	Mat hue_difference = abs(h - hue) * HUE_WEIGHT;
	Mat sat_difference = abs(s - saturation);
	Mat val_difference = max(v - value, 0);

	Mat total = hue_difference + sat_difference + val_difference;
	Mat difference;
	total.convertTo(difference, CV_8U);

	return difference;
}

vector<Vec3f> FindBalls(const Mat& image, const Mat& perspective_matrix)
{
	int min_rad, max_rad;
	BallRadiusRange(perspective_matrix, min_rad, max_rad);

	Mat difference = DifferenceFromCloth(image, perspective_matrix);
	double cloth_thresh = Otsu(difference, TableMask(image.size(), perspective_matrix));

	Mat blurred;
	GaussianBlur(difference, blurred, Size(9,9), 2);

	vector<Vec3f> circles;
	HoughCircles(blurred, circles, HOUGH_GRADIENT, 1, 2 * min_rad,
				cloth_thresh, OUTLINE_FRACTION * 2 * CV_PI * min_rad, min_rad, max_rad);

	Mat not_cloth;
	threshold(difference, not_cloth, cloth_thresh, 255, THRESH_BINARY);

	vector<Vec3f> balls;
	for (const Vec3f& found : circles)
	{
		vector<Point2f> table_position = ToTableCoordinates(vector<Point2f>{Point2f(found[0], found[1])}, perspective_matrix);
		if (table_position[0].x < 0 || table_position[0].x >= TABLE_WIDTH ||
				table_position[0].y < 0 || table_position[0].y >= TABLE_HEIGHT)
			continue;

		Mat disc = Mat::zeros(image.size(), CV_8UC1);
		circle(disc, Point((int)found[0], (int)found[1]), (int)found[2], Scalar(255), FILLED);
		double fill = (double)countNonZero(disc & not_cloth) / countNonZero(disc);
		if (fill >= 0.5)
			balls.push_back(found);
	}
	return balls;
}

const int NUMBER_OF_BALLS = 16;
const int GROUND_TRUTH[7][NUMBER_OF_BALLS][2] = {
	{ // AllBalls.jpeg
		{253, 908}, {250, 959}, {255, 1013}, {256, 1072}, {254, 1132}, {262, 1184}, {256, 1244}, {253, 1307},
		{329, 901}, {330, 961}, {335, 1013}, {336, 1067}, {334, 1130}, {345, 1187}, {342, 1245}, {293, 797}
	},
	{ // Table1.jpeg
	                {309, 1092}, {429, 1212}, {383, 1131}, {335, 1050}, {285, 1211}, {261, 1170}, {352, 1173}, {331, 1132},
					{240, 1210}, {310, 1170}, {407, 1169}, {281, 1135}, {374, 1212}, {360, 1091}, {331, 1213}, {288, 29}
	},
	{ // Table2.jpeg
	                {282, 1103}, {620, 1224}, {391, 1121}, {315, 1046}, {310, 1305}, {202, 621}, {361, 1173}, {331, 1132},
					{177, 725}, {185, 1269}, {665, 798}, {296, 438}, {53, 1235}, {362, 1090}, {399, 1205}, {507, 1169}
	},
	{ // Table3.jpeg
	                {23, 908}, {616, 1224}, {447, 1199}, {336, 29}, {305, 1406}, {210, 621}, {260, 1294}, {226, 1252},
					{177, 725}, {171, 1270}, {665, 798}, {249, 351}, {53, 1233}, {169, 901}, {399, 1201}, {434, 29}
	},
	{ // Table4.jpeg
	                {23, 908}, {546, 1407}, {444, 1204}, {-1, -1}, {305, 1406}, {210, 621}, {260, 1294}, {226, 1252},
					{181, 724}, {171, 1270}, {665, 798}, {249, 351}, {58, 1231}, {169, 901}, {399, 1201}, {88, 1392}
	},
	{ // Table5.jpeg
	                {535, 23}, {546, 1407}, {444, 1204}, {-1, -1}, {305, 1406}, {131, 25}, {260, 1294}, {226, 1252},
					{538, 72}, {171, 1270}, {665, 798}, {249, 351}, {-1, -1}, {169, 901}, {399, 1201}, {608, 26}
	},
	{ // Table6.jpeg
	                {535, 23}, {477, 1364}, {444, 1204}, {-1, -1}, {305, 1406}, {131, 25}, {260, 1294}, {226, 1252},
					{538, 72}, {171, 1270}, {-1, -1}, {249, 351}, {-1, -1}, {169, 901}, {399, 1201}, {629, 947}
	}
};

vector<Point2f> GetGroundTruth(int image_idx)
{
	vector<Point2f> positions;
	for (int ball = 0; ball < NUMBER_OF_BALLS; ball++)
	{
		int row = GROUND_TRUTH[image_idx][ball][0];
		int col = GROUND_TRUTH[image_idx][ball][1];

		if (row >= 0)
			positions.push_back(Point2f((float)col, (float)row));
	}

	return positions;
}

int FoundBalls(const vector<Point2f>& detected, const vector<Point2f>& truth, double max_distance)
{
	int found = 0;
	for (const Point2f& true_pos : truth)
		for (const Point2f& position : detected)
			if (norm(position - true_pos) <= max_distance)
			{
				found ++;
				break;
			}

	return found;
}

BallFeat GetFeatures(const Mat& image, const Vec3f& ball)
{
	Point center((int)ball[0], (int)ball[1]);

	Mat inner = Mat::zeros(image.size(), CV_8UC1);
	circle(inner, center, (int)(ball[2] + 0.7), Scalar(255), FILLED);

	Mat outer = Mat::zeros(image.size(), CV_8UC1);
	circle(outer, center, (int)(ball[2] * RING_OUTER), Scalar(255), FILLED);
	circle(outer, center, (int)(ball[2] * RING_INNER), Scalar(0), FILLED);

	Mat hsv;
	cvtColor(image, hsv, COLOR_BGR2HSV);
	Mat white;
	inRange(hsv, Scalar(0, 0, WHITE_MIN_VAL), Scalar(180, WHITE_MAX_SATURATION, 255), white);

	Mat colored = inner & ~white;

	BallFeat features;
	features.white = (double)countNonZero(white & outer) / countNonZero(outer);
	if (countNonZero(colored) > 30)
		features.color = mean(image, colored);
	else
		features.color = mean(image, inner);

	return features;
}

vector<BallFeat> TrainBalls(const Mat& all_balls, const Mat& perspective_matrix)
{
	vector<BallFeat> trained;
	Mat inverse = perspective_matrix.inv();

	for (int ball = 0; ball < NUMBER_OF_BALLS; ball++)
	{
		Point2f table_positions((float)GROUND_TRUTH[0][ball][1],(float)GROUND_TRUTH[0][ball][0]);

		vector<Point2f> photo_position;
		perspectiveTransform(vector<Point2f>{table_positions}, photo_position, inverse);
		float radius = BallRadiusInPhoto(table_positions, perspective_matrix);

		trained.push_back(GetFeatures(all_balls, Vec3f(photo_position[0].x, photo_position[0].y, radius)));
	}

	return trained;
}

int ClassifyBall(const BallFeat& features, const vector<BallFeat>& trained)
{
	int best = 0;
	double best_distance = 1e18;

	for (int i = 0; i < (int)trained.size(); i++)
	{
		double distance = pow(WHITE_WEIGHT * (features.white - trained[i].white), 2);
		for (int channel = 0; channel < 3; channel++)
			distance += pow(features.color[channel] - trained[i].color[channel], 2);

		if (distance < best_distance)
		{
			best_distance = distance;
			best = i;
		}
	}

	return best;
}

int TrueBallLabel(int image_idx, const Point2f& position, double max_distance)
{
	int label = -1;
	double best = max_distance;

	for (int ball = 0; ball < NUMBER_OF_BALLS; ball++)
	{
		if (GROUND_TRUTH[image_idx][ball][0] < 0)
			continue;

		Point2f truth_position((float)GROUND_TRUTH[image_idx][ball][1], (float)GROUND_TRUTH[image_idx][ball][0]);
		double distance = norm(position - truth_position);
		if (distance <= best)
		{
			best = distance;
			label = ball;
		}
	}

	return label;
}
struct Counts
{
	long tp = 0, fp = 0, fn = 0, tn = 0;

	void Add(const Counts& c)
	{
		tp += c.tp;
		fp += c.fp;
		fn += c.fn;
		tn += c.tn;
	}
};

Mat BallMap(const vector<Point2f>& positions)
{
	Mat map = Mat::zeros(TABLE_HEIGHT, TABLE_WIDTH, CV_8UC1);
	for (const Point2f& position : positions)
		circle(map, position, (int)BALL_RADIUS, Scalar(255), FILLED);

	return map;
}

Counts CompareMaps(const vector<Point2f>& detected, const vector<Point2f>& truth)
{
	Mat found = BallMap(detected);
	Mat real = BallMap(truth);

	Counts counts;
	counts.tp = countNonZero(found & real);
	counts.fp = countNonZero(found & ~real);
	counts.fn = countNonZero(~found & real);
	counts.tn = countNonZero(~found & ~real);

	return counts;
}

void PrintCounts(const string& name, const Counts& c)
{
	printf("%-12s recall %5.1f%%  precision %5.1f%%  accuracy %5.2f%%  specificity %5.2f%%\n", name.c_str(),
			100.0 * c.tp / (c.tp + c.fn), 100.0 * c.tp / (c.tp + c.fp),
			100.0 * (c.tp + c.tn) / (c.tp + c.tn + c.fp + c.fn), 100.0 * c.tn / (c.tn + c.fp));
}

void MyApplication()
{
	// Images are loaded from the Media/Pool folder
	char *file_location = "Media/Pool/";
	char *pool_files[] = {
		"AllBalls.jpeg",
		"Table1.jpeg",
		"Table2.jpeg",
		"Table3.jpeg",
		"Table4.jpeg",
		"Table5.jpeg",
		"Table6.jpeg"};

	int number_of_images = sizeof(pool_files) / sizeof(pool_files[0]);

	Mat perspective_matrix = GetPerspectiveMatrix();

	Mat all_balls = imread(string(file_location) + "AllBalls.jpeg", IMREAD_COLOR);
	vector<BallFeat> trained = TrainBalls(all_balls, perspective_matrix);

	int total_found = 0;
	int total_balls = 0;
	int total_correct = 0;
	int total_matched = 0;
	Counts total_counts;

	namedWindow("detections", WINDOW_NORMAL);
	resizeWindow("detections", 1600, 420);

	for (int file_no = 0; file_no < number_of_images; file_no++)
	{
		string filename(file_location);
		filename.append(pool_files[file_no]);
		Mat input = imread(filename, IMREAD_COLOR);
		if (input.empty())
		{
			cout << "Could not open " << filename << endl;
			continue;
		}

		// ---- YOUR PROCESSING GOES HERE ----

		vector<Vec3f> balls = FindBalls(input, perspective_matrix);

		//Mat detections = input.clone();
		// vector<Point2f> centers;
		// for (const Vec3f& ball : balls)
		// {
		// 	circle(detections, Point((int)ball[0], (int)ball[1]), (int)ball[2], Scalar(0, 0, 255), 2);
		// 	circle(detections, Point((int)ball[0], (int)ball[1]), 3, Scalar(255, 0, 255), FILLED);
		// 	centers.push_back(Point2f(ball[0], ball[1]));
		// }

		vector<Point2f> photo_centers;
		for (const Vec3f& ball : balls)
			photo_centers.push_back(Point2f(ball[0], ball[1]));

		vector<Point2f> detected = ToTableCoordinates(photo_centers, perspective_matrix);

		vector<Point2f> truth = GetGroundTruth(file_no);
		int found = FoundBalls(detected, truth, BALL_RADIUS);
		Counts counts = CompareMaps(detected, truth);

		Mat detections = input.clone();
		int correct = 0;
		int matched = 0;
		for (int i = 0; i < (int)balls.size(); i++)
		{
			int predicted = ClassifyBall(GetFeatures(input, balls[i]), trained);
			int true_label = TrueBallLabel(file_no, detected[i], BALL_RADIUS);

			if (true_label >= 0)
			{
				matched++;
				if (predicted == true_label)
					correct++;
			}

			Point center((int)balls[i][0], (int)balls[i][1]);
			circle(detections, center, (int)balls[i][2], Scalar(0, 0, 255), 2);
			putText(detections, predicted == 15 ? "W" : to_string(predicted + 1), center + Point(-8, 6),
        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 2);
		}

		// -----------------------------------

		//cout << pool_files[file_no] << ": " << input.cols << "x" << input.rows << endl;
		//printf("%s: found %d of %d balls (%.1f%%)\n", pool_files[file_no], found, (int)truth.size(), 100.0 * found / truth.size());
		printf("%s: found %d of %d balls (%.1f%%), labelled %d of %d detected balls correctly\n",
					   pool_files[file_no], found, (int)truth.size(), 100.0 * found / truth.size(), correct, matched);
		PrintCounts("   map:", counts);

		if (file_no > 0)
		{
			total_found += found;
			total_balls += (int)truth.size();
			total_correct += correct;
			total_matched += matched;
			total_counts.Add(counts);
		}

		// Mat side_by_side = JoinImagesHorizontally(input, "Input", output, "Output", 4);
		// imshow("My Application", side_by_side);

		// namedWindow("detections", WINDOW_NORMAL);
		// resizeWindow("detections", 1600, 420);
		imshow("detections", detections);

		char key = waitKey(); // press any key for the next image, 'x' to quit
		if (key == 'x' || key == 'X')
			break;
	}
	destroyAllWindows();
	//printf("Tables 1-6: found %d of %d balls (%.1f%%)\n", total_found, total_balls, 100.0 * total_found / total_balls);
	printf("tables 1-6 detection:   found %d of %d balls (%.1f%%)\n", total_found, total_balls, 100.0 * total_found / total_balls);
	printf("tables 1-6 recognition: %d of %d detected balls labelled correctly (%.1f%%)\n", total_correct, total_matched, 100.0 * total_correct / total_matched);
	printf("tables 1-6 end to end:  %d of %d balls found and labelled correctly (%.1f%%)\n", total_correct, total_balls, 100.0 * total_correct / total_balls);
}
