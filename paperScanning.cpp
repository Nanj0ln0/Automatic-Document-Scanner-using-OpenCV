#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>


using namespace cv;
using namespace std;

const char input[] = "input_picture";

//��ͼ
Mat grabCutDemo(Mat &image,Mat &cutImage);
//��ת
Mat rotatePaper(Mat & image, Mat & rotateImage);
//��ͼ
Mat cutPaper(Mat& image, Mat& cutImage);



int main() {

	Mat src = imread("D:/OpenCV/picture zone/paper.jpg");  //paper.jpg   paper-rotate.png
	if (src.empty()) 
	{
		printf("ERROR");
		return -1;
	}
	namedWindow(input,CV_WINDOW_AUTOSIZE);
	imshow(input,src);
	
	//��ͼ
	Mat result;
	grabCutDemo(src, result);


	//Canny��Ե��� + ������ȡ
	
	//�ղ���
	Mat close;
	Mat k = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	morphologyEx(result, close, MORPH_CLOSE, k, Point(-1, -1), 7);

	//ģ��+canny��Ե���
	Mat gray, Gblur, Gcanny;
	cvtColor(close, gray, COLOR_BGR2GRAY);
	GaussianBlur(gray, Gblur, Size(11, 11), 0, 0);
	Canny(Gblur, Gcanny, 0, 200, 3, false);

	//�������
	vector<vector<Point>> contours;
	vector<Vec4i>hierachy;
	findContours(Gcanny, contours, hierachy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	//��������
	Mat drawImage = Mat::zeros(src.size(),CV_8UC3);
	for (size_t i = 0; i < contours.size(); i++)
	{
		drawContours(drawImage,contours,i,Scalar(0,0,255),3,8,hierachy,0,Point(0,0));
	}
	imshow("contours", drawImage);
	
	Mat grayc;
	cvtColor(drawImage, grayc,COLOR_BGR2GRAY);
	
	vector<Point2f>Features;   //shi-tomasi  �ǵ�����
	double maxCorners = 4;
	double quilitylevel = 0.01;
	double minDistance = 10;
	double blockSize = 3;
	goodFeaturesToTrack(gray, Features, maxCorners, quilitylevel, minDistance, Mat(), blockSize, false, 0.04);

	
	for (size_t t = 0; t < Features.size(); t++)
	{
		circle(drawImage, Features[t], 2, Scalar(255, 255, 255), 2, 8, 0);
	}
	imshow("Features", drawImage);
	
	for (int i = 0; i < 4; i++)
	{
		printf("x=%f,t=%f\n", Features[i].x, Features[i].y);
	}

	
	Point left_top = Features[0];		//���Ͻǵĵ��ܺ���С
	Point right_top = Features[0];	//���Ͻǵĵ������С
	Point left_bottom = Features[0];	//���½ǵĵ�������
	Point right_bottom = Features[0];	//���½ǵĵ��ܺ����

	int a[4];
	a[0] = Features[0].x + Features[0].y;
	a[1] = Features[1].x + Features[1].y;
	a[2] = Features[2].x + Features[2].y;
	a[3] = Features[3].x + Features[3].y;
	
	int Nmin = a[0];
	int Nmax = a[0];
	for (int i = 0; i < 4; i++)
	{
		if (Nmin > a[i])  //�ҳ��ܺ���С�ģ����Ͻǣ�
		{
			Nmin = a[i];
			left_top = Features[i];
		}
		
		if (Nmax < a[i])  //�ҳ��ܺ����ģ����½ǣ�
		{
			Nmax = a[i];
			right_bottom = Features[i];
		}
	}
	
	//�ų�ƥ���2��
	 Point2f DFeature[4];
	for (size_t i = 0; i < Features.size(); i++)
	{
		if (Features[i].x == left_top.x || Features[i].x == right_bottom.x)
		{
			DFeature[i] = Point2f(0.0, 0.0);
		}
		else
		{
			DFeature[i] = Features[i];
		}
	}

	
	int b[4];
	b[0] = abs(DFeature[0].x - DFeature[0].y);
	b[1] = abs(DFeature[1].x - DFeature[1].y);
	b[2] = abs(DFeature[2].x - DFeature[2].y);
	b[3] = abs(DFeature[3].x - DFeature[3].y);
	
	int c;
	Point aa;
	//���½ǵĵ�������
	//���Ͻǵĵ������С
	for (int i = 0; i < 4; i++)
	{
		if (b[i] == 0)
		{
			continue;
		}
		else
		{
			c = b[i];
			right_top = Features[i];
			aa = Features[i];
			break;
		}
	}
	for (int i = 0; i < 4; i++)
	{
		if (b[i] == 0)
		{
			continue;
		}
		else
		{
			if (c == b[i])
			{
				continue;
			}
			
			if (c < b[i])
			{
				left_bottom = Features[i];
			}
			if (c > b[i])
			{
				right_top = Features[i];
				left_bottom = aa;
			}
		}
	}

	//͸�ӱ任
		//�õ�4����
	vector<Point2f>  src_corners(4);
	src_corners[0] = Point(left_top.x, left_top.y); //����
	src_corners[1] = Point(right_top.x , right_top.y);	//����
	src_corners[2] = Point(left_bottom.x, left_bottom.y);	//����
	src_corners[3] = Point(right_bottom.x,right_bottom.y);	//����

	vector<Point2f> dst_corners(4);
	dst_corners[0] = Point(0, 0);
	dst_corners[1] = Point(src.cols, 0);
	dst_corners[2] = Point(0, src.rows);
	dst_corners[3] = Point(src.cols, src.rows);


	// ��ȡ͸�ӱ任����
	Mat outputImage;
	Mat warpmatrix = getPerspectiveTransform(src_corners, dst_corners);
	warpPerspective(src, outputImage, warpmatrix, outputImage.size(), INTER_LINEAR);

	imshow("result", outputImage);

	/*
	//��ת
	Mat dst_rotate;
	rotatePaper(result, dst_rotate);
	imshow("dst_rotate", dst_rotate);
	

	//��ͼ
	//Mat outputImage;
	//cutPaper(dst_rotate, outputImage);
	//namedWindow("Final Result", CV_WINDOW_AUTOSIZE);
	//imshow("Final Result", outputImage);
	*/

	waitKey(0);
	return 0;
}


Mat grabCutDemo(Mat& image, Mat& cutImage)
{

	//�û�ѡ������
	Rect selection;
	Rect2d first = selectROI(input, image);
	selection.x = first.x;
	selection.y = first.y;
	selection.width = first.width;
	selection.height = first.height;
	printf("ROI.x = %d \n ROI.y = %d \n  width = %d \n  height = %d \n", selection.x, selection.y, selection.width, selection.height);

	//grabCut����׼��
	Mat mask, bgModel, fgmodel;//mask���򣬱�����ǰ��
	mask.create(image.size(), CV_8UC1);
	mask.setTo(Scalar::all(GC_BGD));
	mask(selection).setTo(Scalar(GC_PR_FGD));
	grabCut(image, mask, selection, bgModel, fgmodel, 1, GC_INIT_WITH_RECT);

	//��ʾ��������
	Mat binMask;
	binMask.create(mask.size(), CV_8UC1);
	binMask = mask & 1;
	image.copyTo(cutImage, binMask);

	return cutImage;
}

Mat rotatePaper(Mat& image, Mat& rotateImage)
{

	//�ղ��� ׼��grabCut
	Mat close;
	Mat k = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	morphologyEx(image, close, MORPH_CLOSE, k, Point(-1, -1), 5);
	
	//ģ��+canny��Ե���
	Mat gray, Gblur, Gcanny;
	cvtColor(close, gray, COLOR_BGR2GRAY);
	GaussianBlur(gray, Gblur, Size(11, 11), 0, 0);
	Canny(Gblur, Gcanny, 50, 200, 3, false);

	//�������
	vector<vector<Point>> contours;
	vector<Vec4i>hierachy;
	findContours(Gcanny, contours, hierachy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	Mat drawImg = Mat::zeros(close.size(), CV_8UC3);

	float maxw = 0;
	float maxh = 0;
	double degree_rotate;
	for (size_t t = 0; t < contours.size(); t++)
	{
		RotatedRect minRect = minAreaRect(contours[t]);  //��ת���Σ���ȡ��С��ת����
		degree_rotate = abs(minRect.angle);

		//����нǶȣ����������Ⱥ͸߶�
		if (degree_rotate > 0)
		{
			maxw = max(maxw, minRect.size.width);
			maxh = max(maxh, minRect.size.height);
		}
	}

	printf("angle:%f", degree_rotate);

	for (size_t t = 0; t < contours.size(); t++)
	{
		RotatedRect minRect = minAreaRect(contours[t]);
		if ((maxw == minRect.size.width) && (maxh == minRect.size.height))
		{
			degree_rotate = minRect.angle;
			Point2f pts[4];
			minRect.points(pts);
			for (int i = 0; i < 4; i++)
			{
				line(drawImg, pts[i], pts[(i + 1) % 4], Scalar(255, 255, 255), 2, 8, 0);
			}

		}
	}

	Point2f center(image.cols / 2, image.rows / 2);
	Mat rotm = getRotationMatrix2D(center, degree_rotate, 1.0);

	//��ת
	Mat dst_rotate;
	warpAffine(image, rotateImage, rotm, image.size(), INTER_LINEAR, 0, Scalar(0, 0,0));
	return rotateImage;
}

Mat cutPaper(Mat& image, Mat& cutImage)
{
	//�û�ѡ������
	Rect selection;
	Rect2d first = selectROI("dst_rotate", image);
	selection.x = first.x;
	selection.y = first.y;
	selection.width = first.width;
	selection.height = first.height;
	printf("ROI.x = %d \n ROI.y = %d \n  width = %d \n  height = %d \n", selection.x, selection.y, selection.width, selection.height);

	//͸�ӱ任
		//�õ�4����
	vector<Point2f>  src_corners(4);
	src_corners[0] = Point(selection.x, selection.y); //����
	src_corners[1] = Point(selection.x + selection.width, selection.y);	//����
	src_corners[2] = Point(selection.x, selection.y + selection.height);	//����
	src_corners[3] = Point(selection.x + selection.width, selection.y + selection.height);	//����

	vector<Point2f> dst_corners(4);
	dst_corners[0] = Point(0, 0);
	dst_corners[1] = Point(image.cols, 0);
	dst_corners[2] = Point(0, image.rows);
	dst_corners[3] = Point(image.cols, image.rows);

	// ��ȡ͸�ӱ任����
	Mat warpmatrix = getPerspectiveTransform(src_corners, dst_corners);
	warpPerspective(image, cutImage, warpmatrix, cutImage.size(), INTER_LINEAR);

	return cutImage;
}
