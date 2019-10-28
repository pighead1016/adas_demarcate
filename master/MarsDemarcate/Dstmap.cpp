#include "stdafx.h"
#include "Dstmap.h"
#include "line.h"
#include <opencv2/opencv.hpp>
#include "camera_calibration.h"
extern adas_camera jp6_camera;

void gammaarr(float gammanum)
{
	for (int i = 0; i < 255; ++i)
	{
		float temp=i/255.0;
		float a = powf(temp, gammanum);

		gammaarray[i] = int(a*255.0);
	}
}
//生成
void CreateDstmap(int centerx, int centery, int radis, int srcHeigth, int srcWidth, int dstHeigth, int dstWidth)
{
	gammaarr(3.0);	
	//%为畸变创建一维的查询索引映射数组
	//%Input：
	//%centerx: 中心点行坐标
	//%centery: 中心点列坐标
	//%radis  ： 内园半径
	//%srcHeigth：源图像高度
	//%srcWidth： 源图像宽度
	//%dstHeigth：目标图像高度
	//%dstWidth： 目标图像宽度
	//%确定外圆半径
	int a1, a2, outRadis;
	if (centery < srcHeigth / 2)
		a1 = centery;
	else
		a1 = srcHeigth - centery;

	if (centerx < srcWidth / 2)
		a2 = centerx;
	else
		a2 = srcWidth - centery;

	if (a1 < a2)
		outRadis = a1;
	else
		outRadis = a2;

	float thea, tempx, xup, yup, xdown, ydown;
	int temp1, temp2, indexup, indexdown;
	int dstHeiHalf = dstHeigth / 2;
	for (int ii = 0; ii<dstWidth; ii++)
	{
		thea = 180.0f / dstWidth*ii;
		thea = thea*M_PI / 180;

		for (int aa = 0; aa < dstHeiHalf; aa++)
		{
			tempx = radis + (outRadis - radis)*(aa + 1) / dstHeiHalf;
			yup = centery - tempx*sin(thea);
			xup = centerx - tempx*cos(thea);
			ydown = centery + tempx*sin(thea);
			xdown = centerx - tempx*cos(thea);
			temp1 = int(yup + 0.5);
			temp2 = int(xup + 0.5);
			//indexup = temp2*srcHeigth+ temp1;
			indexup = temp2 + temp1*srcWidth;
			g_visualTemplate[ii + (dstHeiHalf - aa - 1)*dstWidth] = indexup;

			temp1 = int(ydown + 0.5) - 1;
			temp2 = int(xdown + 0.5) - 1;
			// indexdown = temp2*srcHeigth+ temp1 + dstHeiHalf;
			indexdown = temp2 + temp1*srcWidth;
			g_visualTemplate[ii + (dstHeigth - aa - 1)*dstWidth] = indexdown;
		} //end aa
	}//end ii
	/*
	for (int i = 0; i < jp6_camera._lane_width*jp6_camera._lane_height; ++i)
	{
		int j = i%jp6_camera._lane_width *2 + jp6_camera._lane_x *2 + (i / jp6_camera._lane_width *2 + jp6_camera._lane_y*2)*videoWidth ;
		g_visualline[i] = g_visualTemplate[j];
	}*/
	/*for (int i = 0; i < face_width*face_height; ++i)
	{
		int j = i%face_width + facecut_x + (i / face_width + facecut_y+ videoHeight/2)*videoWidth ;
		g_visualface[i] = g_visualTemplate[j];
	}*/
}
//转换
int Undist(IplImage *img, IplImage *dst,int width,int height,int x,int y,int up_down)
{
	if (g_visualTemplate == NULL || img == NULL || dst == NULL)
	{
		return -1;
	}

	for (int i = 0; i < width*height; ++i)
	{
		//int x = i%cut_width;
		int yy = i / width;

		int j = (i-yy*width*2)+ x*2 + (yy+ y*2)*videoWidth/2;
		//int j = i%jp6_camera._lane_width * 2 + jp6_camera._lane_x * 2 + (i / jp6_camera._lane_width * 2 + jp6_camera._lane_y * 2)*videoWidth;
		*(dst->imageData + i) = uchar(*(img->imageData + g_visualTemplate[j + videoHeight*videoWidth*up_down / 2]));
		//*(dst->imageData + i * 3 + 0) = *(img->imageData + g_visualTemplate[j + videoHeight*videoWidth*up_down / 2] * 3 + 0);
		//*(dst->imageData + i * 3 + 1) = *(img->imageData + g_visualTemplate[j + videoHeight*videoWidth*up_down / 2] * 3 + 1);
		//*(dst->imageData + i * 3 + 2) = *(img->imageData + g_visualTemplate[j + videoHeight*videoWidth*up_down / 2] * 3 + 2);
	}
	return 0;
}
/*void Undistface(IplImage *img, IplImage *dst)
{
	if (g_visualface == NULL || img == NULL || dst == NULL)
	{
		return;
	}
	for (int i = 0; i < face_width*face_height; ++i)
	{

		*(dst->imageData + i) = *(img->imageData + g_visualface[i]);
	}
}
void Undistline(IplImage *img, IplImage *dst)
{
	for (int i = 0; i < jp6_camera._lane_width*jp6_camera._lane_height; ++i)
	{
		*(dst->imageData + i) = gammaarray[uchar(*(img->imageData + g_visualline[i]))];//int(f*f*f*255);
	}
	
}*/

int Undist_line(IplImage *img, IplImage *dst,int _lane_width,int _lane_height,int _lane_x,int _lane_y)
{
	if (g_visualTemplate == NULL || img == NULL || dst == NULL)
	{
		return -1;
	}
	for (int i = 0; i < _lane_width*_lane_height; ++i)
	{
		int x = i%_lane_width;
		int y = i / _lane_width;
		int j = (x + _lane_x) * 2 + (y + _lane_y) * 2 * videoWidth;
		*(dst->imageData + i) = /*gammaarray[*/uchar(*(img->imageData + g_visualTemplate[j] /*g_visualline[i]*/))/*]*/;//int(f*f*f*255);
	}
}
int Undistall(IplImage *img, IplImage *dst)
{
	

	for (int i = 0; i < videoHeight*videoWidth; ++i)
	{
		//int x = i%cut_width;
		//int yy = i / width;

		//int j = (i - yy*width * 2) + x * 2 + (yy + y * 2)*videoWidth / 2;
		//int j = i%jp6_camera._lane_width * 2 + jp6_camera._lane_x * 2 + (i / jp6_camera._lane_width * 2 + jp6_camera._lane_y * 2)*videoWidth;
		
		*(dst->imageData + i * 3 + 0) = *(img->imageData + g_visualTemplate[i] * 3 + 0);
		*(dst->imageData + i * 3 + 1) = *(img->imageData + g_visualTemplate[i] * 3 + 1);
		*(dst->imageData + i * 3 + 2) = *(img->imageData + g_visualTemplate[i] * 3 + 2);
	}
	return 0;
}
