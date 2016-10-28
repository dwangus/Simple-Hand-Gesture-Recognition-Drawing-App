/*
*	CS440 Spring 2016
*	Programming Assignment 1
*	Team Members: Cyril Saade, Rebecca Jellinek, Ivan Uvarov, David Wang
*/

#include "stdafx.h"

//opencv libraries
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

//C++ standard libraries
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

int myMax(int a, int b, int c);
int myMin(int a, int b, int c);
void mySkinDetect(Mat& src, Mat& dst);
void myFrameDifferencing(Mat& first, Mat& sec, Mat& destination);
void myMotionEnergy(Mat *totalarr, Mat& dest, int size);
void myFrameSum(Mat& drawingFrame, Mat& handFrame, Mat& destination);

double dist(Point x, Point y)
{
	return (x.x - y.x)*(x.x - y.x) + (x.y - y.y)*(x.y - y.y);
}

pair<Point, double> circleFromPoints(Point p1, Point p2, Point p3)
{
	double offset = pow(p2.x, 2) + pow(p2.y, 2);
	double bc = (pow(p1.x, 2) + pow(p1.y, 2) - offset) / 2.0;
	double cd = (offset - pow(p3.x, 2) - pow(p3.y, 2)) / 2.0;
	double det = (p1.x - p2.x) * (p2.y - p3.y) - (p2.x - p3.x)* (p1.y - p2.y);
	double TOL = 0.0000001;
	if (abs(det) < TOL) { cout << "POINTS TOO CLOSE" << endl; return make_pair(Point(0, 0), 0); }

	double idet = 1 / det;
	double centerx = (bc * (p2.y - p3.y) - cd * (p1.y - p2.y)) * idet;
	double centery = (cd * (p1.x - p2.x) - bc * (p2.x - p3.x)) * idet;
	double radius = sqrt(pow(p2.x - centerx, 2) + pow(p2.y - centery, 2));

	return make_pair(Point(centerx, centery), radius);
}

//Can play around with:
//	- Threshold value for frame-differencing
//	- Threshold for minimum area of a contour to be recognized
//	- Blur Filter
//	- Bounding rectangle for a given contour
// 	- # of Convex Hull Points necessary to be recognized
//	- Depth threshold for convexity-defect to be recognized



int main()
{
	// open the video camera no. 0
	VideoCapture cap(0);

	// if not successful, exit program
	if (!cap.isOpened())
	{
		cout << "Cannot open the video cam" << endl;
		return -1;
	}

	// Mat object to read first frame
	Mat frame0;

	// read a new frame from video
	bool bSuccess0 = cap.read(frame0);

	//if not successful, break loop
	if (!bSuccess0)
	{
		cout << "Cannot read a frame from video stream" << endl;
		cap.release();
		return 0;
	}

	bool drawingIsOn = false; // Variable that determines if drawing mode is on or not
	Point previousPoint; //stores the center of the palm of the hand of the previous frame
	bool previousPointIsSet = false;
	Mat drawingFrame; //frame that will hold all lines that have been drawn



	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	Mat first = frame0;
	// Mat second;
	vector<pair<Point, double>> palm_centers;
	

	while (1)
	{
		// read a new frame from video
		Mat frame;
		bool bSuccess = cap.read(frame);

		if (!bSuccess)
		{
			cout << "Cannot read a frame from video stream" << endl;
			break;
		}

		Mat handForDrawingFrame = Mat::zeros(frame.rows, frame.cols, CV_8UC3); //In drawing mode, this frame will display the face or the hand of the user
		Mat drawingDispay = Mat::zeros(frame.rows, frame.cols, CV_8UC3); //this is the frame of the drawing mode

		// In order to create the frame of the drawing mode, it super-imposes the handForDrawingFrame and the drawingFrame
		
		flip(frame, frame, 1);

		Mat skinfilter = Mat::zeros(frame.rows, frame.cols, CV_8UC1);

		mySkinDetect(frame, skinfilter); // // PROCESSING THE IMAGE: we are applying our skinDetection algorithm to the original frame

		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		findContours(skinfilter, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
		// Finding all contours inside the skinfiltered image
		//Stored within the "contours" vector are vectors of points, each vector of points a contour-outline found in the input_frame

		Mat finalDest = Mat::zeros(frame.size(), CV_8UC3);

		//Find contour in vector w/ largest area
		int maxSize = 0;
		int maxIndex = 0;
		//Do we even really need a bounding rectangle?
		//Rect boundrec;

		//Finding greatest contour (storing its index and its area)
		for (int i = 0; i < contours.size(); i++){
			double area = contourArea(contours[i]);
			if (area > maxSize) { //can add another if-condition to screen out only areas of a certain pixel number... Wait, no, that's redundant, since we're already only
				//	screening out the biggest blob on the screen
				maxSize = area;
				maxIndex = i;
				//boundrec = boundingRect(contours[i]);
			}
		}

		vector<vector<Point>> hulls(contours.size());
		vector<vector<int>> hullsIndex(contours.size());
		//each convexity-defect part of a set of convexity-defects corresponding to a convex hull + contour 
		//		is represented as a 4-element integer vector (start_index, end_index, farthest_point_index, fixpoint_depth);
		//		fixpoint_depth is an approx. of the distance between the farthest contour point and the convex hull
		//SO, basically, each defect is, between two fingers, basically a trace of the tip of one finger down and up back to 
		//		the next, with the start_index the beginning of the trace in the original contour, the end_index the end,
		//		the farthest_point_index the farthest point to the convex hull in the original contour, within the defect,
		//		and the fixpoint_depth the distance of that farthest point
		vector<vector<Vec4i>> defects(contours.size());

		for (int i = 0; i < contours.size(); i++){
			convexHull(contours[i], hulls[i]);
			convexHull(contours[i], hullsIndex[i]);
			if (hullsIndex[i].size() > 7){ //you can play with the number of the size
				convexityDefects(contours[i], hullsIndex[i], defects[i]);
			}
		}
		vector<Vec4i> largestDefect = defects[maxIndex];
		int thresh_area = 100; // can plan around with the lowest minimum threshold area
		int convexNum = 0;
		if (maxSize > thresh_area){
			//Scalar(B,G,R)
			drawContours(frame, hulls, maxIndex, Scalar(0, 255, 0), 1, 8, hierarchy);
			drawContours(handForDrawingFrame, hulls, maxIndex, Scalar(0, 255, 0), 1, 8, hierarchy);
			for (int i = 0; i < largestDefect.size(); i++){
				const Vec4i& vecDef = largestDefect[i];
				float pointDepth = vecDef[3] / 256.0;
				if (pointDepth > 50){	//can play around with recognized depth
					Point farthest(contours[maxIndex][(vecDef[2])]);
					circle(frame, farthest, 6, Scalar(0, 0, 255), 3);
					//circle(handForDrawingFrame, farthest, 6, Scalar(0, 0, 255), 3);
					convexNum++;
				}
			}
		}
		int no_of_fingers = 0;

		Point rough_palm_center;
		if (largestDefect.size() >= 3){
			vector<Point> palm_points;
			for (int j = 0; j<largestDefect.size(); j++)
			{
				int startidx = largestDefect[j][0]; Point ptStart(contours[maxIndex][startidx]);
				int endidx = largestDefect[j][1]; Point ptEnd(contours[maxIndex][endidx]);
				int faridx = largestDefect[j][2]; Point ptFar(contours[maxIndex][faridx]);
				//Sum up all the hull and defect points to compute average
				rough_palm_center += ptFar + ptStart + ptEnd;
				palm_points.push_back(ptFar);
				palm_points.push_back(ptStart);
				palm_points.push_back(ptEnd);
			}
			//Get palm center by 1st getting the average of all defect points, this is the rough palm center,
			//Then you chose the closest 3 points and get the circle radius and center formed from them which is the palm center.
			rough_palm_center.x /= largestDefect.size() * 3;
			rough_palm_center.y /= largestDefect.size() * 3;
			Point closest_pt = palm_points[0];
			vector<pair<double, int> > distvec;
			for (int i = 0; i<palm_points.size(); i++)
				distvec.push_back(make_pair(dist(rough_palm_center, palm_points[i]), i));
			sort(distvec.begin(), distvec.end());
			//Keep choosing 3 points till you find a circle with a valid radius
			//As there is a high chance that the closest points might be in a linear line or too close that it forms a very large circle
			pair<Point, double> soln_circle;
			for (int i = 0; i + 2<distvec.size(); i++)
			{
				Point p1 = palm_points[distvec[i + 0].second];
				Point p2 = palm_points[distvec[i + 1].second];
				Point p3 = palm_points[distvec[i + 2].second];
				soln_circle = circleFromPoints(p1, p2, p3);//Final palm center,radius
				if (soln_circle.second != 0)
					break;
			}
			//Find avg palm centers for the last few frames to stabilize its centers, also find the avg radius
			palm_centers.push_back(soln_circle);
			if (palm_centers.size()>10)
				palm_centers.erase(palm_centers.begin());
			Point palm_center;
			double radius = 0;
			for (int i = 0; i<palm_centers.size(); i++)
			{
				palm_center += palm_centers[i].first;
				radius += palm_centers[i].second;
			}
			palm_center.x /= palm_centers.size();
			palm_center.y /= palm_centers.size();
			radius /= palm_centers.size();
			//Draw the palm center and the palm circle
			//The size of the palm gives the depth of the hand
			circle(frame, palm_center, 5, Scalar(144, 144, 255), 3);
			circle(handForDrawingFrame, palm_center, 5, Scalar(144, 144, 255), 3);

			if (previousPointIsSet) {

				Scalar lineColor;
				putText(handForDrawingFrame, "Hold 'D' to quit.", Point(30, 30), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255));
				putText(handForDrawingFrame, "Open hand to draw in white, close hand to draw in orange.", Point(30, 50), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255));
				
				if (convexNum < 3) { //if a fist is detected
					lineColor = Scalar(11, 113, 251);

					putText(handForDrawingFrame, "Fist detected, you are drawing in orange", Point(30, 70), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255));
				}
				else { //if more than 3 fingers are detected
					lineColor = Scalar(255, 255, 255);

					putText(handForDrawingFrame, "You are drawing in white", Point(30, 70), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255));
				}

				line(drawingFrame, previousPoint, palm_center, lineColor, 5);
				
			}
			previousPoint = palm_center;
			previousPointIsSet = true;

			
			//Detect fingers by finding points that form an almost isosceles triangle with certain thesholds
			for (int j = 0; j<largestDefect.size(); j++){
				int startidx = largestDefect[j][0]; Point ptStart(contours[maxIndex][startidx]);
				int endidx = largestDefect[j][1]; Point ptEnd(contours[maxIndex][endidx]);
				int faridx = largestDefect[j][2]; Point ptFar(contours[maxIndex][faridx]);
				//X o--------------------------o Y
				double Xdist = sqrt(dist(palm_center, ptFar));
				double Ydist = sqrt(dist(palm_center, ptStart));
				double length = sqrt(dist(ptFar, ptStart));
				double retLength = sqrt(dist(ptEnd, ptFar));
				//Play with these thresholds to improve performance
				if (length <= 3 * radius&&Ydist >= 0.4*radius&&length >= 10 && retLength >= 10 && max(length, retLength) / min(length, retLength) >= 0.8){
					if (min(Xdist, Ydist) / max(Xdist, Ydist) <= 0.8)
					{
						if ((Xdist >= 0.1*radius&&Xdist <= 1.3*radius&&Xdist<Ydist) || (Ydist >= 0.1*radius&&Ydist <= 1.3*radius&&Xdist>Ydist))
							line(frame, ptEnd, ptFar, Scalar(0, 255, 0), 1), no_of_fingers++;
					}
				}
			}
			no_of_fingers = min(5, no_of_fingers);
		}

		drawContours(frame, contours, maxIndex, Scalar(255, 0, 0), 1, 8, hierarchy);
		drawContours(handForDrawingFrame, contours, maxIndex, Scalar(255, 0, 0), 1, 8, hierarchy);

		putText(frame, "Num of Fingers: " + to_string(convexNum), Point(30, 70), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 255));
		putText(frame, "Welcome! Hold 'D' to start drawing.", Point(30, 30), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 255));


		imshow("Main View", frame);

		if (drawingIsOn) {
			myFrameSum(drawingFrame, handForDrawingFrame, drawingDispay);
			imshow("Drawing", drawingDispay);
		}


		//esc key pressed by user
		if (waitKey(30) == 27)
		{
			cout << "Escaped" << endl;
			break;
		}

		if (waitKey(30) == 'd' && !drawingIsOn) { // D key pressed by the user (to enter drawing mode)

			imshow("Drawing", drawingDispay);
			cout << "Entering drawing mode" << endl;
			drawingIsOn = true;
			drawingFrame = Mat::zeros(frame.rows, frame.cols, CV_8UC3);
			cout << "Started the drawing game" << endl;
			previousPointIsSet = false;
		}

		if (waitKey(30) == 'd' && drawingIsOn) {
			drawingIsOn = false;
			destroyWindow("Drawing");
			cout << "Exiting drawing mode" << endl;
		}

	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


	cap.release();
	return 0;
}


void myFrameSum(Mat& drawingFrame, Mat& handFrame, Mat& destination) { // For the drawing mode, it super-imposes the face frame and the line that are being drawn by the user
	for (int i = 0; i < drawingFrame.rows; i++){
		for (int j = 0; j < drawingFrame.cols; j++){

			Vec3b drawingFP = drawingFrame.at<Vec3b>(i, j);
			Vec3b handFP = handFrame.at<Vec3b>(i, j);

			int drawingAve = (drawingFP[0] + drawingFP[1] + drawingFP[2]) / 3;
			int handAve = (handFP[0] + handFP[1] + handFP[2]) / 3;

			if (drawingAve == 0 && handAve == 0) {
				// black pixel
				destination.at<Vec3b>(i, j).val[0] = 0;
				destination.at<Vec3b>(i, j).val[1] = 0;
				destination.at<Vec3b>(i, j).val[2] = 0;
			}
			else if (drawingAve != 0 && handAve == 0) {
				// show drawing pixel
				destination.at<Vec3b>(i, j) = drawingFP;
			}
			else {
				// show face pixel
				destination.at<Vec3b>(i, j) = handFP;
			}
		}		
	}
}
//Function that returns the maximum of 3 integers
int myMax(int a, int b, int c) {
	return max(max(a, b), c);
}
//Function that returns the minimum of 3 integers
int myMin(int a, int b, int c) {
	return min(min(a, b), c);
}
//Function that detects whether a pixel belongs to the skin based on RGB values
void mySkinDetect(Mat& src, Mat& dst) {
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			Vec3b intensity = src.at<Vec3b>(i, j);
			int blue = intensity[0]; // red
			int green = intensity[1]; // green
			int red = intensity[2]; // blue

			if ((red > 95) && (blue > 20) && (green > 40) && ((myMax(red, green, blue) - myMin(red, green, blue)) > 15) && (abs(red - green) > 15) && (red > green) && (red > blue)) {
				dst.at<uchar>(i, j) = 255;
			}

		}
	}

}
//Function that does frame differencing between the current frame and the previous frame
void myFrameDifferencing(Mat& first, Mat& sec, Mat& destination) {
	for (int i = 0; i < first.rows; i++){
		for (int j = 0; j < first.cols; j++){
			Vec3b firstI = first.at<Vec3b>(i, j);
			Vec3b secI = sec.at<Vec3b>(i, j);
			int ave1 = (firstI[0] + firstI[1] + firstI[2]) / 3;
			int ave2 = (secI[0] + secI[1] + secI[2]) / 3;
			int diff = ave2 - ave1;
			int ssd = diff*diff;
			int thresh = 1000;
			if (ssd >= thresh){
				destination.at<Vec3b>(i, j) = firstI;
			}
			else{
				destination.at<Vec3b>(i, j).val[0] = 0;
				destination.at<Vec3b>(i, j).val[1] = 0;
				destination.at<Vec3b>(i, j).val[2] = 0;

			}
		}
	}
}