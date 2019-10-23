#pragma once
#define _USE_MATH_DEFINES
#include <opencv2/opencv.hpp>
#include <math.h>
#include "line.h"
#define videoWidth 1632//2048
#define videoHeight 1632//1536

#define facecut_x 900  //600
#define facecut_y 120  //100
#define face_width 264	//264	
#define face_height 264

#define dst_scaling 0.5 
#define up 0
#define down 1
static int gammaarray[256];
static int* g_visualTemplate = new int[videoWidth*videoHeight];
static int* g_visualline = new int[videoWidth*videoHeight];
//static int* g_visualface = new int[face_width*face_height];
void CreateDstmap(int centerx, int centery, int radis, int srcHeigth, int srcWidth, int dstHeigth, int dstWidth);
int Undist(IplImage *img, IplImage *dst, int width, int height, int x, int y, int up_down=0);
void Undistline(IplImage *img, IplImage *dst);
int Undistall(IplImage *img, IplImage *dst);
//void Undistface(IplImage *img, IplImage *dst);
//void gammaarr(float gammanum);
