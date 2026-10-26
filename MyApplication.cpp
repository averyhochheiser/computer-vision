// Entry point for your application (menu option 'm')

#include "Utilities.h"
#include "src/ImageLoader.h"

void MyApplication()
{
	// // Images are loaded from the Media/Pool folder
	// char *file_location = "Media/Pool/";
	// char *pool_files[] = {
	// 	"AllBalls.jpeg",
	// 	"Table1.jpeg",
	// 	"Table2.jpeg",
	// 	"Table3.jpeg",
	// 	"Table4.jpeg",
	// 	"Table5.jpeg",
	// 	"Table6.jpeg"};
	//
	// int number_of_images = sizeof(pool_files) / sizeof(pool_files[0]);
	//
	// for (int file_no = 0; file_no < number_of_images; file_no++)
	// {
	// 	string filename(file_location);
	// 	filename.append(pool_files[file_no]);
	// 	Mat input = imread(filename, IMREAD_COLOR);
	// 	if (input.empty())
	// 	{
	// 		cout << "Could not open " << filename << endl;
	// 		continue;
	// 	}
	//
	// 	// ---- YOUR PROCESSING GOES HERE ----
	// 	Mat output = input.clone();
	// 	// -----------------------------------
	//
	// 	cout << pool_files[file_no] << ": " << input.cols << "x" << input.rows << endl;
	//
	// 	Mat side_by_side = JoinImagesHorizontally(input, "Input", output, "Output", 4);
	// 	imshow("My Application", side_by_side);
	//
	// 	char key = waitKey(); // press any key for the next image, 'x' to quit
	// 	if (key == 'x' || key == 'X')
	// 		break;
	// }
	// destroyAllWindows();

	vector<Mat> images;
	vector<string> names;

	if (LoadImages("Media/Pool/", images, names) == 0)
	{
		return;
	}

	for (size_t i = 0; i < images.size(); i++)
	{
		// preprocessing
		imshow(names[i], images[i]);

		char key = waitKey();
		destroyAllWindows();
		if (key == 'x')
			break;
	}
}
