/*
 * DenseTrajectory.hh
 *
 *  Created on: May 8, 2017
 *      Author: ahsan
 */

#ifndef ACTIONRECOGNITION_DENSETRAJECTORY_HH_
#define ACTIONRECOGNITION_DENSETRAJECTORY_HH_
#include "Utils.hh"
#include "Math/Matrix.hh"
#include "Math/Vector.hh"
#include <list>
#include "Gmm.hh"

struct Point {
	f32 x, y, t;
	Point(f32 _x, f32 _y, f32 _t) { x = _x; y = _y; t = _t; }
	Point(const Point& p) {
		this->x = p.x;
		this->y = p.y;
		this->t = p.t;
	}
	static Point getDifference(const Point &p1, const Point &p2) {
		return Point(p1.x - p2.x, p1.y - p2.y, p2.t);
	}
};

struct Trajectory {
	std::vector<Point> track;

	bool isMaxLengthAchieved;

	Math::Vector<f32> hog;
	Math::Vector<f32> hof;
	Math::Vector<f32> mbhX;
	Math::Vector<f32> mbhY;
	Math::Vector<f32> trajectoryShape;
	Math::Vector<f32> finalDesc;

	Trajectory(): track(), isMaxLengthAchieved(false), hog(8*2*2*15), hof(9*2*2*15), mbhX(8*2*2*15), mbhY(8*2*2*15), trajectoryShape(30), finalDesc(426) {
		hog.fill(0.0f); hof.fill(0.0f); mbhX.fill(0.0f); mbhY.fill(0.0f); trajectoryShape.fill(0.0f);
		finalDesc.fill(0.0);
	}
	virtual ~Trajectory() {}
	void addPoint(Point& p) { track.push_back(p); }
	u32 getSize() { return track.size(); }
	Point& getLastPoint() {
		if (track.size() == 0)
			Core::Error::msg("dense-trajectory.trajectory is empty.") << Core::Error::abort;
		return track.at(track.size() - 1);
	}
	void updateTrajectoryShape(f32 deltaX, f32 deltaY, u32 index) {
		trajectoryShape.at(2*index) = deltaX;
		trajectoryShape.at(2*index + 1) = deltaY;
	}
	void normalizeTrajShape() {
		f32 norm = 0;
		for (u32 i=0; i<trajectoryShape.size()/2; i++) {
			norm += sqrt(trajectoryShape.at(2*i) * trajectoryShape.at(2*i) + trajectoryShape.at(2*i+1) * trajectoryShape.at(2*i+1));
		}
		for (u32 i=0; i<trajectoryShape.size(); i++) {
			trajectoryShape.at(i) /= norm;
		}
	}
	void normalize(Math::Vector<f32>& vec, u32 binSize) {
		for (u32 i=0; i<vec.size(); i+=binSize) {
			f32 norm = 0;
			for (u32 j=i; j< (i+binSize); j++) {
				norm += vec.at(j);// * vec.at(j);
			}
			//assert(norm > 0);
			//norm = sqrt(norm);
			for (u32 j=i; j<(i+binSize); j++) {
				vec.at(j) /= norm;
			}
		}
	}

	void finalizeDesc(Math::Vector<f32>& desc, u32 bins, u32 index) {
		u32 tStride = 5;
		u32 dim = bins * 2 * 2;
		u32 pos = 0;
		for(u32 i = 0; i < 3; i++) {
			std::vector<float> vec(dim);
			for(u32 t = 0; t < tStride; t++) {
				for(u32 j = 0; j < dim; j++) {
					vec[j] += desc[pos++];
				}
			}
			for(u32 j = 0; j < dim; j++) {
				finalDesc.at(index + (i * dim) + j) = vec[j]/5.0f;
			}
		}
	}


	void finalizeDescriptor() {
		/*f32 l2norm = trajectoryShape.normEuclidean();
		trajectoryShape.scale(1.0f/l2norm);*/

		normalizeTrajShape();

		finalDesc.copyBlockFromVector(trajectoryShape, 0, 0, trajectoryShape.nRows());
		finalizeDesc(hog, 8, 30);
		finalizeDesc(hof, 9, 126);
		finalizeDesc(mbhX, 8, 234);
		finalizeDesc(mbhY, 8, 330);

		/*normalize(hog, 8);
		normalize(hof, 9);
		normalize(mbhX, 8);
		normalize(mbhY, 8);
		finalDesc.copyBlockFromVector(trajectoryShape, 0, 0, trajectoryShape.nRows());
		finalDesc.copyBlockFromVector(hog, 0, trajectoryShape.nRows(), hog.nRows());
		finalDesc.copyBlockFromVector(hof, 0, trajectoryShape.nRows() + hog.nRows(), hof.nRows());
		finalDesc.copyBlockFromVector(mbhX, 0, trajectoryShape.nRows() + hog.nRows() + hof.nRows(), mbhX.nRows());
		finalDesc.copyBlockFromVector(mbhY, 0, trajectoryShape.nRows() + hog.nRows() + hof.nRows() + mbhX.nRows(), mbhY.nRows());*/
	}
};

class Program {
private:
	static const Core::ParameterInt paramSmaplingWindow_;
	static const Core::ParameterFloat paramQuality_;
	static const Core::ParameterInt paramMedianFilterSize_;
	static const Core::ParameterInt paramMaxTrajectoryLength_;
	static const Core::ParameterFloat paramMinTrajVariance_;

	static const Core::ParameterString paramVideoList_;

	static const Core::ParameterInt paramWindowSize_;
	static const Core::ParameterFloat paramMinOpFlowMag_;
	static const Core::ParameterString paramFeaturesPath_;
	static const Core::ParameterString paramPCAParamPath_;
	static const Core::ParameterString paramPCAFeaturesPath_;
	static const Core::ParameterFloat paramEpsilon_;

	static const Core::ParameterEnum paramTask_;
	enum Task { claculateAndSaveTrajectories, calculatePCA, applyPCA, calculateGMM, calFisherVectors};

	u32 samplingWindow_;
	f32 quality_;
	u32 medianFilterSize_;
	u32 maxTrajectoryLength_;
	f32 minTrajVariance_;
	std::string videoList_;
	u32 windowSize_;
	f32 minOpFlowMag_;
	std::string featuresPath_;
	std::string pcaParamPath_;
	std::string pcaFeaturesPath_;
	f32 epsilon_;

	std::list<Trajectory> trajectories;


	void getRectangleAroundPoint(const Point& p, cv::Rect& result, u32 frameWindth, u32 frameHeight);
	//void calculateHistogram(const cv::Mat& magnitude, const cv::Mat& angle, const cv::Rect& rectangle, std::vector<f32>& result, u32 bins, u32 timeStep, bool isHOF);
	void calculateHistogram(const cv::Mat& magnitude, const cv::Mat& angle, const cv::Rect& rectangle, Math::Vector<f32>& result, u32 bins, u32 timeStep, bool isHOF);
	void calculateMagnitudeAndAngle(const cv::Mat& frame, const cv::Mat& flowX, const cv::Mat& flowY, cv::Mat& mag, cv::Mat& angle, cv::Mat& flowMag,
			cv::Mat& flowAngle, cv::Mat& flowXMag, cv::Mat& flowXAngle, cv::Mat& flowYMag, cv::Mat& flowYAngle);
	void visualizeTrajectories(std::vector<cv::Mat>& video);
	bool isValid(Trajectory& traj);
	void postProcessTrajectories();
	void getPointsAlongTrajectories(std::vector<Point>& points);
	void calculateDerivative(const cv::Mat& image, cv::Mat& xDer, cv::Mat& yDer);
	void trackTrajectories(const cv::Mat& prevFrame, const cv::Mat& flowX, const cv::Mat& flowY);
	void initializeTrajectories(std::vector<Point> &interestPoints);
	void visualizePoints(cv::Mat& frame, std::vector<Point>& points);
	void calculateTrajectories(std::vector<cv::Mat> &video);
	void applyMedianFilter(std::vector<cv::Mat> &flows);
	void calculateOpticalFlow(std::vector<cv::Mat> &video, std::vector<cv::Mat> &flows_x, std::vector<cv::Mat> &flows_y);
	void extractDensePoints(cv::Mat &frame, std::vector<Point> &points);

	void trajectoriesToMatrix(Math::Matrix<f32>& result);



	void calculateAndSaveDenseTrajectories();
	void calculateAndSavePCA();
	void applyPCAAndSave();
	void calculateGMMParams();

	void calculateFisherVectors();

public:
	Program();
	void BuildDescMat(const cv::Mat& mag, const cv::Mat& angle, std::vector<f32>& intHist,
			u32 nBins, bool isHof);
	void GetDesc(const std::vector<f32>& intHist, Math::Vector<f32>& result,
			u32 intHistHeight, u32 intHistWidth, cv::Rect rect, u32 nBins,
			const u32 nxCells, const u32 nyCells, const u32 index);
	virtual ~Program() { }
	void run();
};



#endif /* ACTIONRECOGNITION_DENSETRAJECTORY_HH_ */
