//Sobel算子边缘提取
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace std;
using namespace cv;


void Sobel(Mat& src, Mat& dst)
{
	Mat dst1, dst2;
	Mat getPrewitt_horizontal = (Mat_<float>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
	Mat getPrewitt_vertical = (Mat_<float>(3, 3) << 1, 0, -1, 2, 0, -2, 1, 0, -1); 

	flip(getPrewitt_horizontal, getPrewitt_horizontal, -1);
	flip(getPrewitt_vertical, getPrewitt_vertical, -1);

	filter2D(src, dst1, src.depth(), getPrewitt_horizontal);
	filter2D(src, dst2, src.depth(), getPrewitt_vertical); 

	dst = dst1 + dst2;
}


int main()
{
	Mat srcImage = imread("6.jpg");
	Mat dstImage;

	//判断图像是否加载成功
	if (srcImage.data)
		cout << "图像加载成功!" << endl << endl;
	else
	{
		cout << "图像加载失败!" << endl << endl;
		return -1;
	}
	//转换为灰度图
	Mat grayInput;
	cvtColor(srcImage, grayInput, COLOR_BGR2GRAY);
	Mat out_gray = grayInput;

	namedWindow("srcImage", WINDOW_NORMAL);
	imshow("srcImage", srcImage);

	Sobel(grayInput, dstImage);

	namedWindow("dstImage", WINDOW_NORMAL);
	imshow("dstImage", dstImage);

	waitKey(0);

	return 0;
}

