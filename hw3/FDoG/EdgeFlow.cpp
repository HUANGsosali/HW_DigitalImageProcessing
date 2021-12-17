// testSift.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
//#include "opencv2/core/core.hpp"//因为在属性中已经配置了opencv等目录，所以把其当成了本地目录一样
//#include "opencv2/features2d/features2d.hpp"
//#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
#include <iostream>

#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/imgproc/imgproc.hpp"  

#include "imatrix.h"
#include "ETF.h"
#include "fdog.h"
#include "myvec.h"

using namespace cv;
using namespace std;

void dogFilter(Mat grayImage);

int main(int argc, char* argv[])
{
	string imgPath;
	double tao = 0.99;
	double thres = 0.7;
	double sigma = 1.0;
	double sigma3 = 3.0;

	int wz = 15;
	double colSig = 15.0;
	double spaSig = 10.0;
	int iterFDoG = 2;

	//cout << argc << endl;
	for(int i=1; i<argc; i++)
	{
		if( strcmp(argv[i], "-i") ==0 )
		{
			imgPath = argv[++i];
			continue;
		}
		if( strcmp(argv[i], "-sm") ==0 )
		{
			sigma = atof(argv[++i]);
			continue;
		}
		if( strcmp(argv[i], "-sm3") ==0 )
		{
			sigma3 = atof(argv[++i]);
			continue;
		}
		if( strcmp(argv[i], "-to") ==0 )
		{
			tao = atof(argv[++i]);
			continue;
		}
		if( strcmp(argv[i], "-th") ==0 )
		{
			thres = atof(argv[++i]);
			continue;
		}
	}
	cout << imgPath << "   " << endl;
	Mat image;
	Mat grayImage;
	if(!imgPath.empty())
	{
		image = imread(imgPath, 1);
		grayImage = imread(imgPath, 0);
		if( image.empty() || grayImage.empty())
		{
			cout << "Read image error!" ;
			return -1;
		}
	}
	
	//bilateral filtering
	Mat filterImg;
	//color sigma对降噪起了作用
	bilateralFilter(grayImage, filterImg, wz, colSig, spaSig);
	/*imshow("Edge Image", grayImage);
	imshow("bilateralImage", filterImg);
	waitKey(0);*/
	grayImage = filterImg;

   	cout<<"rows: "<<image.rows<<"   "<<"cols: "<<image.cols<<endl;
	imatrix img;
	int index;

	// 2014-5-6 flag=1 use a,b   flag=0, use I
	img.init(image.rows, image.cols);

	for (int i = 0; i < image.rows; i++) {
			img.p[i] = new int[image.cols];
			for (int j = 0; j < image.cols; j++) {
				index = i*image.cols+j;
				//printf("1: %d\n", grayImage.data[index]);
				img.p[i][j] = grayImage.data[index];
				//printf("2: %d\n", img.p[i][j]);
			}
	}

	int image_x = img.getRow();
	int image_y = img.getCol();
	ETF e;
	e.init(image_x, image_y);
	e.set(img); // get gradients from input image
	//e.set2(img); // get gradients from gradient map
	e.Smooth(4, 2);
	
	GetFDoG(img, e, sigma, sigma3, tao);

	//by me 2014-5-6
	//Iterative FDoG
	int pxVal;
	for (int i = 0; i<iterFDoG; i++)
	{
		for(int j = 0; j < img.getRow(); j++)
		{
			for(int k = 0; k < img.getCol(); k++)
			{
				pxVal = (int)img[j][k] + (int)(grayImage.data[j*img.getCol()+k]);
				if(pxVal > 255)
					pxVal = 255;
				img[j][k] = pxVal;
			}
		}
		GetFDoG(img, e, sigma, sigma3, tao);
	}
	
	GrayThresholding(img, thres); 

	Mat saveImg = grayImage;
	for (int i = 0; i < image.rows; i++) {
			for (int j = 0; j < image.cols; j++) {
				index = i*image.cols+j;
				saveImg.data[index] = img.p[i][j];
			}
	}
	/*imshow("Edge Image", saveImg);
	waitKey(0);*/
	int lastSlashPos = 0;
	string str = "\\";
	for(int i=imgPath.length()-1; i>=0; i--)
	{
		if(imgPath[i] == str[0])
		{
			lastSlashPos = i;
			break;
		}
	}
	
	string basePath = imgPath.substr(0, lastSlashPos);
	string imgName = imgPath.substr(lastSlashPos, imgPath.length()-basePath.length()-4);
	string ext = imgPath.substr(imgPath.length()-4, 4);

	string savePath = basePath+imgName+"_edge"+ext;

	/*cout << "basePath: " << basePath << endl;
	cout << "ImageName: " << imgName << endl;
	cout << "ext: " << ext << endl;
	cout << "output path: " << savePath << endl;*/
	imwrite(savePath, saveImg);
	//system("pause");
    return 0;
}


void dogFilter(Mat grayImage)
{
	// DoG
	Mat img_G0;
	Mat img_G1;
	Mat img_DoG;
	Mat tmpImg1;
	tmpImg1 = grayImage;
	GaussianBlur(tmpImg1,img_G0,Size(5,5),1, 1);
	GaussianBlur(tmpImg1,img_G1,Size(5,5),7, 7);
	img_DoG = img_G0  - img_G1;
	normalize(img_DoG,img_DoG,255,0,CV_MINMAX);  

	imshow("Blur1", img_G0);
	imshow("Blur2", img_G1);
	imshow("DoG", img_DoG);
	waitKey(0);
	/*imwrite("E:\\ori.jpg", grayImage);
	imwrite("E:\\blur1.jpg", img_G0);
	imwrite("E:\\blur2.jpg", img_G1);*/
	imwrite("E:\\DoG.jpg", img_DoG);
}



