#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <queue>
#include <ctime>

using namespace std;
using namespace cv;

// constants

namespace accurate_eye_detect {

	// Debugging
	const bool kPlotVectorField = false;

	// Size constants
	const int kEyePercentTop = 25;
	const int kEyePercentSide = 13;
	const int kEyePercentHeight = 30;
	const int kEyePercentWidth = 35;

	// Preprocessing
	const bool kSmoothFaceImage = false;
	const float kSmoothFaceFactor = 0.005f;

	// Algorithm Parameters
	const int kFastEyeWidth = 50;
	const int kWeightBlurSize = 1;
	const bool kEnableWeight = true;
	const float kWeightDivisor = 1.0f;
	const double kGradientThreshold = 100;

	// Postprocessing
	const bool kEnablePostProcess = true;
	const float kPostProcessThreshold = 0.97f;

	// Eye Corner
	const bool kEnableEyeCorner = false;

	string face_cascade_name = "res/haarcascade_frontalface_alt.xml";
	CascadeClassifier face_cascade;
	RNG jesus(12345);
	Mat skin_histogram = Mat::zeros(Size(256, 256), CV_8UC1);


	void preprocess();
	void detect_and_display_eyes(Mat& frame);

	int main() {
		preprocess();
		VideoCapture capture(0);
		Mat frame;
		if (capture.isOpened()) {
			while (true) {
				capture >> frame;
				flip(frame, frame, 1);
				if (frame.empty()) return cerr << "No captured frame" << endl, 0;
				detect_and_display_eyes(frame);
				int c = waitKey(1);
				if (c == 'c') break;
			}
		}
		return 0;
	}

	void preprocess() {
		Mat frame;
		if (!face_cascade.load(face_cascade_name)) {
			cerr << "Error in face cascade" << endl;
			throw;
		}
	}

	void scale_to_fast_size(const Mat& src, Mat& dst) {
		float scale = 1.0f * kFastEyeWidth / src.cols * src.rows;
		resize(src, dst, Size(kFastEyeWidth, cvRound(scale)));
	}

	Mat matrix_magnitude(Mat& X, Mat& Y) {
		Mat mags(X.rows, X.cols, CV_64F);
		double mn = HUGE_VAL;
		for (int y = 0; y < X.rows; ++y) {
			double *xr = X.ptr<double>(y);
			double *yr = Y.ptr<double>(y);
			double *mr = mags.ptr<double>(y);
			for (int x = 0; x < X.cols; ++x) {
				double gx = xr[x];
				double gy = yr[x];
				double dist = gx * gx + gy * gy;
				mn = min(mn, dist);
				mr[x] = dist;
			}
		}
		return mags;
	}

	Mat compute_matrix_gradient(const Mat& mat) {
		Mat out(mat.rows, mat.cols, CV_64F);
		for (int y = 0; y < mat.rows; ++y) {
			const uchar *mr = mat.ptr<uchar>(y);
			double *outr = out.ptr<double>(y);
			outr[0] = mr[1] - mr[0];
			for (int x = 1; x < mat.cols - 1; ++x)
				outr[x] = (mr[x + 1] - mr[x - 1]) / 2.0;
			outr[mat.cols - 1] = mr[mat.cols - 1] - mr[mat.cols - 2];
		}
		return out;
	}

	void test_possible_centers_formula(int x, int y, const Mat& weight, double gx, double gy, Mat& out) {
		for (int cy = 0; cy < out.rows; ++cy) {
			double *outr = out.ptr<double>(cy);
			const unsigned char *wr = weight.ptr<unsigned char>(cy);
			for (int cx = 0; cx < out.cols; ++cx) {
				if (x == cx && y == cy) {
					continue;
				}
				// create a vector from the possible center to the gradient origin
				double dx = x - cx;
				double dy = y - cy;
				// normalize d
				double mag = hypot(dx, dy);
				dx /= mag;
				dy /= mag;
				// positive dot product
				double dot_product = max(0.0, dx * gx + dy * gy);
				// square and multiply by the weight
				if (kEnableWeight) outr[cx] += dot_product * dot_product * wr[cx] / kWeightDivisor;
				else outr[cx] += dot_product * dot_product;
			}
		}
	}


	double compute_dynamic_threshold(const cv::Mat &mat, double factor) {
		cv::Scalar stdMagnGrad, meanMagnGrad;
		cv::meanStdDev(mat, meanMagnGrad, stdMagnGrad);
		double stdDev = stdMagnGrad[0] / sqrt(mat.rows*mat.cols);
		return factor * stdDev + meanMagnGrad[0];
	}

	Mat flood_kill_edges(Mat& mat) {
		// kill edges to 0 using BFS
		rectangle(mat, cv::Rect(0, 0, mat.cols, mat.rows), 255);
		Mat mask(mat.rows, mat.cols, CV_8U, 255);
		static int dx[] = {-1, 0, 1, 0};
		static int dy[] = {0, -1, 0, 1};
		queue<Point> q;
		Point origin(0, 0);
		if (mat.at<float>(origin) != 0.0f) {
			q.push(origin);
			mat.at<float>(origin) = 0.0f;
			mask.at<uchar>(origin) = 0;
		}
		while (!q.empty()) {
			Point u = q.front(); q.pop();
			if (mat.at<float>(u) == 0.0f) continue;
			mat.at<float>(u) = 0.0f;
			// add in every direction
			for (int k = 0; k < 4; ++k) {
				int nx = u.x + dx[k];
				int ny = u.y + dy[k];
				Point np(nx, ny);
				if (nx >= 0 && nx < mat.rows && ny >= 0 && ny < mat.cols && mat.at<float>(np) != 0.0f) {
					q.push(np);
					mat.at<float>(np) = 0.0f;
					mask.at<uchar>(np) = 0;
				}
			}
		}
		return mask;
	}

	Point unscale_point(Point p, Rect scale) {
		int x = cvRound(1.0f * p.x / kFastEyeWidth * scale.width);
		int y = cvRound(1.0f * p.y / kFastEyeWidth * scale.width);
		return Point(x, y);
	}

	Point translate_point(const Point& p, const Rect& rect) {
		return Point(p.x + rect.x, p.y + rect.y);
	}

	Point find_eye_center(Mat& face, const Rect& eye) {
		Mat eyeROI_unscaled = face(eye);
		Mat eyeROI;
		scale_to_fast_size(eyeROI_unscaled, eyeROI);
		Mat gradient_x = compute_matrix_gradient(eyeROI);
		Mat gradient_y = compute_matrix_gradient(eyeROI.t()).t();

		// normalize threshold
		Mat mags = matrix_magnitude(gradient_x, gradient_y);
		double gradient_thresh = compute_dynamic_threshold(mags, kGradientThreshold);

		// normalize
		for (int y = 0; y < eyeROI.rows; ++y) {
			double *xr = gradient_x.ptr<double>(y);
			double *yr = gradient_y.ptr<double>(y);
			double *mr = mags.ptr<double>(y);
			for (int x = 0; x < eyeROI.cols; ++x) {
				double gx = xr[x];
				double gy = yr[x];
				double mag = mr[x];
				if (mag > gradient_thresh) {
					xr[x] = gx / mag;
					yr[x] = gy / mag;
				} else {
					xr[x] = 0.0;
					yr[x] = 0.0;
				}
			}
		}


		// create a blurred and inverted image for weighting
		Mat weight;
		cv::GaussianBlur(eyeROI, weight, Size(kWeightBlurSize, kWeightBlurSize), 0, 0);
		for (int y = 0; y < weight.rows; ++y) {
			unsigned char *row = weight.ptr<unsigned char>(y);
			for (int x = 0; x < weight.cols; ++x) {
				row[x] = 255 - row[x];
			}
		}
		
		// run the algorithm
		Mat sum = Mat::zeros(eyeROI.rows, eyeROI.cols, CV_64F);

		priority_queue<pair<double, pair<int, int>>> pq;

		for (int y = 0; y < weight.rows; ++y) {
			const double *xr = gradient_x.ptr<double>(y);
			const double *yr = gradient_y.ptr<double>(y);
			double *mr = mags.ptr<double>(y);
			for (int x = 0; x < weight.cols; ++x) {
				double gx = xr[x];
				double gy = yr[x];
				if (gx == 0.0 && gy == 0.0)
					continue;
				pq.push({-mr[y], {x, y}});
				if (pq.size() > 30) pq.pop();
				// test_possible_centers_formula(x, y, weight, gx, gy, sum);
			}
		}

		while (!pq.empty()) {
			int x = pq.top().second.first;
			int y = pq.top().second.second;
			test_possible_centers_formula(x, y, weight, gradient_x.at<double>(y, x), gradient_y.at<double>(y, x), sum);
			pq.pop();
		}

		// average all values
		double num_gradients = weight.rows * weight.cols;
		Mat out;
		sum.convertTo(out, CV_32F, 1.0 / num_gradients);

		// get the maximum point
		Point maxP;
		double maxVal = -HUGE_VAL;
		cv::minMaxLoc(out, NULL, &maxVal, NULL, &maxP);

		// flood fill the edges
		if (kEnablePostProcess) {
			Mat floodClone;
			double flood_thresh = maxVal * kPostProcessThreshold;
			threshold(out, floodClone, flood_thresh, 0.0f, CV_THRESH_TOZERO);
			Mat mask = flood_kill_edges(floodClone);
			cv::minMaxLoc(out, NULL, &maxVal, NULL, &maxP, mask);
		}

		return unscale_point(maxP, eye);

	}

	pair<Point, Point> find_eyes(Mat& out, Mat& frame, const Rect& face) {
		Mat faceROI = frame(face);
		if (kSmoothFaceImage) {
			double sigma = kSmoothFaceFactor * face.width;
			cv::GaussianBlur(faceROI, faceROI, Size(0, 0), sigma);
		}
		// find eye regions
		float eye_region_width = face.width * kEyePercentWidth / 100.0f;
		float eye_region_height = face.height * kEyePercentHeight / 100.0f;
		float eye_region_top = face.height * kEyePercentTop / 100.0f;
		float eye_region_left = face.width * kEyePercentSide / 100.0f;
		Rect left_eye_region(cvRound(eye_region_left), cvRound(eye_region_top), cvRound(eye_region_width), cvRound(eye_region_height));
		Rect right_eye_region(cvRound(face.width - eye_region_width - eye_region_left), cvRound(eye_region_top), cvRound(eye_region_width), cvRound(eye_region_height));

		Rect adj_left(left_eye_region.x + face.x, left_eye_region.y + face.y, left_eye_region.width, left_eye_region.height);
		Rect adj_right(right_eye_region.x + face.x, right_eye_region.y + face.y, right_eye_region.width, right_eye_region.height);

		rectangle(out, adj_left, Scalar(0, 255, 0), 1);
		rectangle(out, adj_right, Scalar(0, 255, 0), 1);

		// find eye centers
		Point left_pupil = translate_point(translate_point(find_eye_center(faceROI, left_eye_region), left_eye_region), face);
		Point right_pupil = translate_point(translate_point(find_eye_center(faceROI, right_eye_region), right_eye_region), face);
		return {left_pupil, right_pupil};
	}

	void detect_and_display_eyes(Mat& frame) {

		unsigned h = clock();

		Mat output_image = frame.clone();
		vector<Rect> faces;

		// apply filters

		cvtColor(frame, frame, COLOR_BGR2GRAY);
		// equalizeHist(frame, frame);
		medianBlur(frame, frame, 3);

		// detect faces
		int flags = 0
			| CV_HAAR_SCALE_IMAGE
			| CV_HAAR_FIND_BIGGEST_OBJECT;
		Size min_face_size = cv::Size(150, 150);
		face_cascade.detectMultiScale(frame, faces, 1.4, 3, flags, min_face_size);

		// show what you got
		if (faces.size() > 0) {
			auto eyes = find_eyes(output_image, frame, faces[0]);
			line(output_image, eyes.first + Point(0, -10), eyes.first + Point(0, 10), Scalar(0, 255, 0), 1);
			line(output_image, eyes.second + Point(0, -10), eyes.second + Point(0, 10), Scalar(0, 255, 0), 1);
			line(output_image, eyes.first + Point(-10, 0), eyes.first + Point(10, 0), Scalar(0, 255, 0), 1);
			line(output_image, eyes.second + Point(-10, 0), eyes.second + Point(10, 0), Scalar(0, 255, 0), 1);
		}

		float t = (float)(clock() - h) / CLOCKS_PER_SEC * 1000;
		printf("%.2fms\n", t);


		imshow("webcam", output_image);

	}
}

int main() { accurate_eye_detect::main(); }