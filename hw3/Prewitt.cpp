//Prewitt���ӱ�Ե��ȡ
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace std;
using namespace cv;


void prewitt(Mat& src, Mat& dst)
{
	Mat getPrewitt_horizontal = (Mat_<float>(3, 3) << -1, -1, -1, 0, 0, 0, 1, 1, 1); //ˮƽ����
	Mat getPrewitt_vertical = (Mat_<float>(3, 3) << -1, 0, 1, -1, 0, 1, -1, 0, 1); //��ֱ����

	flip(getPrewitt_horizontal, getPrewitt_horizontal, -1);
	flip(getPrewitt_vertical, getPrewitt_vertical, -1);

	Mat dst1, dst2, dst3, dst4;
	filter2D(src, dst1, src.depth(), getPrewitt_horizontal); 
	filter2D(src, dst2, src.depth(), getPrewitt_vertical); 


	dst = dst1 + dst2;
}


int main()
{
	Mat srcImage = imread("6.jpg");
	Mat dstImage;

	//�ж�ͼ���Ƿ���سɹ�
	if (srcImage.data)
		cout << "ͼ����سɹ�!" << endl << endl;
	else
	{
		cout << "ͼ�����ʧ��!" << endl << endl;
		return -1;
	}

	//ת��Ϊ�Ҷ�ͼ
	Mat grayInput;
	cvtColor(srcImage, grayInput, COLOR_BGR2GRAY);
	Mat out_gray = grayInput;

	namedWindow("GrayImage", WINDOW_NORMAL);
	imshow("GrayImage", out_gray);

	prewitt(out_gray, dstImage);

	namedWindow("dstImage", WINDOW_NORMAL);
	imshow("dstImage", dstImage);

	waitKey(0);

	return 0;
}

