#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>

using namespace std;
using namespace cv;

#define PI 3.1415926535 


void get_guassion_array( Mat& kernel,int kernel_size = 3, double sigma = 1)
{
	int center = kernel_size / 2;
	double sum = 0;

	kernel = Mat(kernel_size, kernel_size, CV_32FC1);
	float s = 2 * sigma * sigma;
	for (int i = 0; i < kernel_size; i++)
	{
		for (int j = 0; j < kernel_size; j++)
		{
			int x = i - center;
			int y = j - center;

			kernel.at<float>(i, j) = (1 / (2 * PI * s)) * exp(-((x * x + y * y) / s));
			sum += kernel.at<float>(i, j);
		}
	}

	for (int i = 0; i < kernel_size; i++)
	{
		for (int j = 0; j < kernel_size; j++)
		{
			kernel.at<float>(i, j) /= sum;
		}
	}
	return;
}


void get_gradient_direction(Mat& InputImage, Mat& gradXY, Mat& theta)
{
	int Prewitt_x[3][3] = { 1, 0, -1, 2, 0, -2, 1, 0, -1 }; //垂直
	int Prewitt_y[3][3] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 }; //水平
	gradXY = Mat::zeros(InputImage.size(), CV_32SC1);
	theta = Mat::zeros(InputImage.size(), CV_32SC1);

	int step = InputImage.step;

	uchar* P = InputImage.data;

	for (int i = 1; i < InputImage.rows - 1; i++)
	{
		for (int j = 1; j < InputImage.cols - 1; j++)
		{
			int a[3][3];
			a[0][0] = P[(i - 1) * step + j - 1];
			a[0][1] = P[(i - 1) * step + j];
			a[0][2] = P[(i - 1) * step + j + 1];
			a[1][0] = P[i * step + j - 1];
			a[1][1] = P[i * step + j];
			a[1][2] = P[i * step + j + 1];
			a[2][0] = P[(i + 1) * step + j - 1];
			a[2][1] = P[(i + 1) * step + j];
			a[2][2] = P[(i + 1) * step + j + 1];

			double gradX = 0; 
			double gradY = 0; 
			for (int h = 0; h <= 2; h++)
			{
				for (int k = 0; k <= 2; k++)
				{
					gradX += a[h][k] * Prewitt_x[h][k];
					gradY += a[h][k] * Prewitt_y[h][k];
				}
			}

			theta.at<int>(i, j) = atan(gradY / gradX);
			gradXY.at<int>(i, j) = sqrt(gradX * gradX + gradY * gradY);
		}

	}

	convertScaleAbs(gradXY, gradXY);
}


void non_max_suppression(Mat& imageInput, Mat& imageOutput,  Mat& theta)
{
	imageOutput = imageInput.clone();


	int cols = imageInput.cols;
	int rows = imageInput.rows;

	for (int i = 1; i < rows - 1; i++)
	{
		for (int j = 1; j < cols - 1; j++)
		{
			if (0 == imageInput.at<uchar>(i, j))
				continue;

			int g00 = imageInput.at<uchar>(i - 1, j - 1);
			int g01 = imageInput.at<uchar>(i - 1, j);
			int g02 = imageInput.at<uchar>(i - 1, j + 1);

			int g10 = imageInput.at<uchar>(i, j - 1);
			int g11 = imageInput.at<uchar>(i, j);
			int g12 = imageInput.at<uchar>(i, j + 1);

			int g20 = imageInput.at<uchar>(i + 1, j - 1);
			int g21 = imageInput.at<uchar>(i + 1, j);
			int g22 = imageInput.at<uchar>(i + 1, j + 1);

			int g1 = 0;
			int g2 = 0;
			int g3 = 0;
			int g4 = 0;
			double tmp1 = 0.0;
			double tmp2 = 0.0;

			int direction = theta.at<int>(i, j); 
			double weight = tan(direction);

			if (weight == 0)
			{
				weight = 0.0000001;
			}
			if (weight > 1)
			{
				weight = 1 / weight;
			}
			if ((0 <= direction && direction < 45) || 180 <= direction && direction < 225)
			{
				tmp1 = g10 * (1 - weight) + g20 * (weight);
				tmp2 = g02 * (weight)+g12 * (1 - weight);
			}
			if ((45 <= direction && direction < 90) || 225 <= direction && direction < 270)
			{
				tmp1 = g01 * (1 - weight) + g02 * (weight);
				tmp2 = g20 * (weight)+g21 * (1 - weight);
			}
			if ((90 <= direction && direction < 135) || 270 <= direction && direction < 315)
			{
				tmp1 = g00 * (weight)+g01 * (1 - weight);
				tmp2 = g21 * (1 - weight) + g22 * (weight);
			}
			if ((135 <= direction && direction < 180) || 315 <= direction && direction < 360)
			{
				tmp1 = g00 * (weight)+g10 * (1 - weight);
				tmp2 = g12 * (1 - weight) + g22 * (weight);
			}

			if (imageInput.at<uchar>(i, j) < tmp1 || imageInput.at<uchar>(i, j) < tmp2)
			{
				imageOutput.at<uchar>(i, j) = 0;
			}
		}
	}

}


void double_threshold(Mat& iamgeInput,  double threshold_down,  double threshold_top)
{
	for (int i = 0; i < iamgeInput.rows; i++)
	{
		for (int j = 0; j < iamgeInput.cols; j++) 
		{
			double temp = iamgeInput.at<uchar>(i, j);
			if (temp >= threshold_top)
			{
				temp = 255;
			}
			else if (temp <= threshold_down)
			{
				temp = 0;
			}
			iamgeInput.at<uchar>(i, j) = temp;
		}
	}
}


void hysteresis_threshold(Mat& imageInput, double threshold_down, double threshold_top)
{

	for (int i = 1; i < imageInput.rows - 1; i++)
	{
		for (int j = 1; j < imageInput.cols - 1; j++)
		{
			double pix = imageInput.at<uchar>(i, j);
			if (pix != 255)		continue;
			bool change = false;
			for (int k = -1; k <= 1; k++)
			{
				for (int u = -1; u <= 1; u++)
				{
					if (k == 0 && u == 0)continue;
					double temp = imageInput.at<uchar>(i + k, j + u);
					if (temp >= threshold_down && temp <= threshold_top)
					{
						imageInput.at<uchar>(i + k, j + u) = 255;
						change = true;
					}
				}
			}
			if (change)
			{
				if (i > 1)i--;
				if (j > 2)j -= 2;

			}
		}
	}

	for (int i = 0; i < imageInput.rows; i++)
	{
		for (int j = 0; j < imageInput.cols; j++)
		{
			if (imageInput.at<uchar>(i, j) != 255)
			{
				imageInput.at<uchar>(i, j) = 0;
			}
		}
	}
}


int main()
{

	Mat IN_image = imread("5.jpg");
	

	//转换为灰度图
	Mat grayInput ;
	cvtColor(IN_image, grayInput, COLOR_BGR2GRAY);
	Mat out_gray =  grayInput;


	//高斯滤波
	Mat gausKernel;
	get_guassion_array(gausKernel);
	Mat gaussian_Image;
	filter2D(grayInput, gaussian_Image, grayInput.depth(), gausKernel);
	Mat out_gaussian = gaussian_Image;
	

	//计算XY方向梯度
	//gradient_image
	Mat gradient_image, theta;
	get_gradient_direction(gaussian_Image, gradient_image, theta);
	Mat out_gradient = gradient_image;


	//对梯度幅值进行非极大值抑制
	Mat localImage;
	non_max_suppression(gradient_image, localImage, theta);
	Mat out_nms = localImage;


	//双阈值算法检测和边缘连接
	double_threshold(localImage, 60, 100);
	hysteresis_threshold(localImage, 60, 100);
	Mat out_threshold = localImage;


	//opencv标准canny
	Mat opencvCanny;
	Canny(grayInput, opencvCanny, 60, 100);
	Mat out_opencv = opencvCanny;

	

	/*namedWindow("input", WINDOW_NORMAL);
	imshow("input", IN_image);

	namedWindow("gray", WINDOW_NORMAL);
	imshow("gray", out_gray);

	namedWindow("gaussian", WINDOW_NORMAL);
	imshow("gaussian", out_gaussian);

	namedWindow("gradient", WINDOW_NORMAL);
	imshow("gradient", out_gradient);

	namedWindow("nms", WINDOW_NORMAL);
	imshow("nms", out_nms);*/

	namedWindow("my canny", WINDOW_NORMAL);
	imshow("my canny", out_threshold); 

	/*namedWindow("opencv canny", WINDOW_NORMAL);
	imshow("opencv canny", out_opencv);*/



	waitKey(0);

	destroyAllWindows();

	return 0;
}



