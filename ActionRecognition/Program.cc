/*
 * DenseTrajectory.cc
 *
 *  Created on: May 8, 2017
 *      Author: ahsan
 */
#include "Program.hh"
#include "Gmm.hh"
#include <sstream>
#include "Features/FeatureWriter.hh"
#include "Features/FeatureReader.hh"



const Core::ParameterInt Program::paramSmaplingWindow_("sampling-window", 5, "dense-trajectory");
const Core::ParameterFloat Program::paramQuality_("quality", 0.001, "dense-trajectory");
const Core::ParameterInt Program::paramMedianFilterSize_("median-filter-size", 5, "dense-trajectory");
const Core::ParameterString Program::paramVideoList_("video-list", "", "dense-trajectory");
const Core::ParameterInt Program::paramMaxTrajectoryLength_("max-trajectory-length", 15, "dense-trajectory");
const Core::ParameterFloat Program::paramMinTrajVariance_("min-trajectory-variance", sqrt(3), "dense-trajectory");
const Core::ParameterInt Program::paramWindowSize_("window-size", 32, "dense-trajectory");
const Core::ParameterFloat Program::paramMinOpFlowMag_("min-optical-flow-mag", 0.4f, "dense-trajectory");
const Core::ParameterString Program::paramFeaturesPath_("features-path", "", "dense-trajectory");
const Core::ParameterString Program::paramPCAParamPath_("pca-result-path", "", "dense-trajectory");
const Core::ParameterString Program::paramPCAFeaturesPath_("pca-features-path", "", "dense-trajectory");
const Core::ParameterFloat Program::paramEpsilon_("epsilon", 0.05f, "dense-trajectory");


const Core::ParameterEnum Program::paramTask_("task",
		"claculateAndSaveTrajectories, calculatePCA, applyPCA, calculateGMM, calFisherVectors","claculateAndSaveTrajectories", "dense-trajectory");

Program::Program() :
	samplingWindow_(Core::Configuration::config(paramSmaplingWindow_)),
	quality_(Core::Configuration::config(paramQuality_)),
	medianFilterSize_(Core::Configuration::config(paramMedianFilterSize_)),
	maxTrajectoryLength_(Core::Configuration::config(paramMaxTrajectoryLength_)),
	minTrajVariance_(Core::Configuration::config(paramMinTrajVariance_)),
	videoList_(Core::Configuration::config(paramVideoList_)),
	windowSize_(Core::Configuration::config(paramWindowSize_)),
	minOpFlowMag_(Core::Configuration::config(paramMinOpFlowMag_)),
	featuresPath_(Core::Configuration::config(paramFeaturesPath_)),
	pcaParamPath_(Core::Configuration::config(paramPCAParamPath_)),
	pcaFeaturesPath_(Core::Configuration::config(paramPCAFeaturesPath_)),
	epsilon_(Core::Configuration::config(paramEpsilon_)){

}

//////////////////////////////////
//////////Original Code///////////

// compute integral histograms for the whole image
void Program::BuildDescMat(const cv::Mat& mag, const cv::Mat& angle, std::vector<f32>& intHist,
		u32 nBins, bool isHof)
{
	float maxAngle = 360.f;
	u32 nDims = nBins;
	// one more bin for hof
	nBins = isHof ? nBins-1 : nBins;
	const float angleBase = float(nBins)/maxAngle;

	int step = (mag.cols+1)*nDims;
	int index = step + nDims;
	for(u32 i = 0; i < (u32)mag.rows; i++, index += nDims) {
		const f32 *magRow = mag.ptr<f32>(i);
		const f32 *angleRow = angle.ptr<f32>(i);

		// summarization of the current line
		std::vector<float> sum(nDims);
		for(u32 j = 0; j < (u32)mag.cols; j++) {

			f32 mag0 = magRow[j];
			f32 mag1;
			u32 bin0, bin1;

			// for the zero bin of hof
			if(isHof && mag0 <= minOpFlowMag_) {
				bin0 = nBins; // the zero bin is the last one
				mag0 = 1.0;
				bin1 = 0;
				mag1 = 0;
			}
			else {
				f32 angle = angleRow[j];
				if(angle >= maxAngle) angle -= maxAngle;

				// split the mag to two adjacent bins
				f32 fbin = angle * angleBase;
				bin0 = cvFloor(fbin);
				bin1 = (bin0 + 1) % nBins;

				mag1 = (fbin - bin0) * mag0;
				mag0 -= mag1;
			}

			sum[bin0] += mag0;
			sum[bin1] += mag1;

			for(u32 m = 0; m < nDims; m++, index++)
				intHist[index] = intHist[index-step] + sum[m];
		}
	}
}

// get a descriptor from the integral histogram
void Program::GetDesc(const std::vector<f32>& intHist, Math::Vector<f32>& result,
		u32 intHistHeight, u32 intHistWidth, cv::Rect rect, u32 nBins,
		const u32 nxCells, const u32 nyCells, const u32 index)
{
	u32 dim = nBins * nxCells * nyCells;//descInfo.dim;

	int xStride = rect.width/nxCells;
	int yStride = rect.height/nyCells;
	int xStep = xStride*nBins;
	int yStep = yStride * intHistWidth * nBins;

	// iterate over different cells
	int iDesc = 0;
	std::vector<f32> vec(dim);
	for(u32 xPos = rect.x, x = 0; x < nxCells; xPos += xStride, x++) {
		for(u32 yPos = rect.y, y = 0; y < nyCells; yPos += yStride, y++) {
			// get the positions in the integral histogram
			//const float* top_left = descMat->desc + (yPos * intHistWidth + xPos) * nBins;
			u32 topLeft = (yPos * intHistWidth + xPos) * nBins;
			//const float* top_right = top_left + xStep;
			u32 topRight = topLeft + xStep;
			//const float* bottom_left = top_left + yStep;
			u32 bottomLeft = topLeft + yStep;
			//const float* bottom_right = bottom_left + xStep;
			u32 bottomRight = bottomLeft + xStep;

			for(u32 i = 0; i < nBins; i++) {
				//f32 sum = bottom_right[i] + top_left[i] - bottom_left[i] - top_right[i];
				f32 sum = intHist[bottomRight + i] + intHist[topLeft + i] - intHist[bottomLeft + i] - intHist[topRight + i];
				vec[iDesc++] = std::max<float>(sum, 0) + epsilon_;
			}
		}
	}

	f32 norm = 0;
	for(u32 i = 0; i < dim; i++)
		norm += vec[i];
	if(norm > 0) norm = 1./norm;

	u32 pos = index * dim;
	//std::cout<<result.nRows()<<"::"<<dim<<"::"<<pos<<std::endl;
	for(u32 i = 0; i < dim; i++) {
		/*if (pos >= result.nRows())
			std::cout<<result.nRows()<<"::"<<pos<<"::"<<index<<"::"<<i<<std::endl;*/
		result.at(pos++) = sqrt(vec[i] * norm);
	}
}

//////////////////////////////////
//////////////////////////////////


void Program::visualizePoints(cv::Mat& frame, std::vector<Point>& points) {
	std::cout<<"count:"<<points.size()<<std::endl;
	for (u32 i = 0; i < points.size(); i++) {
		cv::circle(frame, cv::Point(points.at(i).x, points.at(i).y), 10, cv::Scalar(0, 255, 255));
	}
	cv::imshow("Test", frame);
	cv::waitKey(0);
}

void Program::visualizeTrajectories(std::vector<cv::Mat>& video) {
/*
	for (u32 i=0; i<video.size(); i++) {
		for (u32 j=0; j<trajectories.size(); j++) {
			for (u32 k=0; k<trajectories.at(j).track.size() - 1; k++) {

				s32 x1 = cvFloor(trajectories.at(j).track.at(k).x);
				s32 y1 = cvFloor(trajectories.at(j).track.at(k).y);

				s32 x2 = cvFloor(trajectories.at(j).track.at(k+1).x);
				s32 y2 = cvFloor(trajectories.at(j).track.at(k+1).y);

				cv::line(video.at(i), cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 255));
			}
		}
		cv::imshow("Test", video.at(i));
		cv::waitKey(0);
	}
	*/
}

void Program::initializeTrajectories(std::vector<Point> &interestPoints) {
	for (u32 i=0; i<interestPoints.size(); i++) {

		Trajectory traj;
		traj.addPoint(interestPoints.at(i));
		trajectories.push_back(traj);
	}
}

void Program::getPointsAlongTrajectories(std::vector<Point> &points) {
	points.clear();
	for (std::list<Trajectory>::iterator it = trajectories.begin(); it != trajectories.end(); it++) {
		points.push_back(it->getLastPoint());
		/*if (! it->isMaxLengthAchieved) {
			for (u32 i = 0; i<it->getSize(); i++) {
				points.push_back(it->track.at(i));
			}
		}*/
	}
}

void Program::calculateDerivative(const cv::Mat& image, cv::Mat& xDer, cv::Mat& yDer) {
	cv::Sobel(image, xDer, CV_32FC1, 1, 0, 1);
	cv::Sobel(image, yDer, CV_32FC1, 0, 1, 1);
}

void Program::getRectangleAroundPoint(const Point& p, cv::Rect& result, u32 frameWindth, u32 frameHeight) {
	/*result.x = std::min<s32>(std::max<s32>(cvRound(p.x) - windowSize_/2, 0), frameWindth - windowSize_/2);
	result.y = std::min<s32>(std::max<s32>(cvRound(p.y) - windowSize_/2, 0), frameHeight - windowSize_/2);
	result.width = windowSize_;
	result.height = windowSize_;*/

	int x_min = windowSize_/2;
	int y_min = windowSize_/2;
	int x_max = frameWindth - windowSize_;
	int y_max = frameHeight - windowSize_;

	result.x = std::min<int>(std::max<int>(cvRound(p.x) - x_min, 0), x_max);
	result.y = std::min<int>(std::max<int>(cvRound(p.y) - y_min, 0), y_max);
	result.width = windowSize_;
	result.height = windowSize_;
}

void Program::calculateHistogram(const cv::Mat& magnitude, const cv::Mat& angle, const cv::Rect& rectangle,
		Math::Vector<f32> &result, u32 bins, u32 timeStep, bool isHof) {
	u32 sizeOfBin = 360/(isHof ? (bins-1) : bins);

	for (u32 i=rectangle.y, k=0; i<(u32)(rectangle.y+rectangle.height); i++, k++) {
		const f32* rowMag = magnitude.ptr<f32>(i);
		const f32* rowAngle = angle.ptr<f32>(i);
		for (u32 j=rectangle.x, l=0; j<(u32)(rectangle.x+rectangle.width); j++, l++) {

			u32 index = ((timeStep/5) * (bins*4)) + ((k/(bins*2)) * (bins*2)) + ((l/(bins*2)) * bins);
			if (isHof && rowMag[j] < minOpFlowMag_) {
				result.at(index + bins - 1) += 1;
			}
			result.at(index + (((u32)rowAngle[j] % 360)/sizeOfBin)) += rowMag[j] + 0.05;
		}
	}
}

void Program::calculateMagnitudeAndAngle(const cv::Mat& frame, const cv::Mat& flowX, const cv::Mat& flowY, cv::Mat& mag, cv::Mat& angle, cv::Mat& flowMag,
		cv::Mat& flowAngle, cv::Mat& flowXMag, cv::Mat& flowXAngle, cv::Mat& flowYMag, cv::Mat& flowYAngle) {
	cv::Mat xDer, yDer;

	calculateDerivative(frame, xDer, yDer);
	cv::cartToPolar(xDer, yDer, mag, angle, true);

	cv::cartToPolar(flowX, flowY, flowMag, flowAngle, true);

	calculateDerivative(flowX, xDer, yDer);
	cv::cartToPolar(xDer, yDer, flowXMag, flowXAngle, true);

	calculateDerivative(flowY, xDer, yDer);
	cv::cartToPolar(xDer, yDer, flowYMag, flowYAngle, true);
}

void Program::trackTrajectories(const cv::Mat& prevFrame, const cv::Mat& flowX, const cv::Mat& flowY) {

	cv::Mat mag, angle, flowMag, flowAngle, flowXMag, flowXAngle, flowYMag, flowYAngle;
	calculateMagnitudeAndAngle(prevFrame, flowX, flowY, mag, angle, flowMag, flowAngle, flowXMag, flowXAngle, flowYMag, flowYAngle);

	std::vector<f32> hogIntHist((prevFrame.cols + 1) * (prevFrame.rows + 1) * 8, 0.0f);
	std::vector<f32> hofIntHist((prevFrame.cols + 1) * (prevFrame.rows + 1) * 9, 0.0f);
	std::vector<f32> mbhXIntHist((prevFrame.cols + 1) * (prevFrame.rows + 1) * 8, 0.0f);
	std::vector<f32> mbhYIntHist((prevFrame.cols + 1) * (prevFrame.rows + 1) * 8, 0.0f);

	BuildDescMat(mag, angle, hogIntHist, 8, false);
	BuildDescMat(flowMag, flowAngle, hofIntHist, 9, true);
	BuildDescMat(flowXMag, flowXAngle, mbhXIntHist, 8, false);
	BuildDescMat(flowYMag, flowYAngle, mbhYIntHist, 8, false);

	for (std::list<Trajectory>::iterator it = trajectories.begin(); it != trajectories.end(); ) {
		if (it->getSize() > 0 && it->isMaxLengthAchieved == false) {
			Point p = it->getLastPoint();
			s32 x = std::min<int>(std::max<int>(cvRound(p.x), 0), prevFrame.cols - 1);
			s32 y = std::min<int>(std::max<int>(cvRound(p.y), 0), prevFrame.rows - 1);

			f32 new_x = p.x + flowX.at<f32>(y, x);
			f32 new_y = p.y + flowY.at<f32>(y, x);

			/*std::cout<<i<<"::("<<x<<","<<y<<")->("<<new_x<<","<<new_y<<")"<<std::endl;
			i++;
			++it;
			continue;*/

			if (new_x < 0 || new_x >= prevFrame.cols
					|| new_y < 0 || new_y >= prevFrame.rows) {

				it = trajectories.erase(it);
				continue;
			}

			it->updateTrajectoryShape(flowX.at<f32>(y, x), flowY.at<f32>(y, x), p.t);

			cv::Rect rectangle;
			getRectangleAroundPoint(p, rectangle, prevFrame.cols, prevFrame.rows);
			GetDesc(hogIntHist, it->hog, prevFrame.rows + 1, prevFrame.cols + 1, rectangle, 8, 2, 2, p.t);
			GetDesc(hofIntHist, it->hof, prevFrame.rows + 1, prevFrame.cols + 1, rectangle, 9, 2, 2, p.t);
			GetDesc(mbhXIntHist, it->mbhX, prevFrame.rows + 1, prevFrame.cols + 1, rectangle, 8, 2, 2, p.t);
			GetDesc(mbhXIntHist, it->mbhY, prevFrame.rows + 1, prevFrame.cols + 1, rectangle, 8, 2, 2, p.t);

			/*calculateHistogram(mag, angle, rectangle, it->hog, 8, p.t, false);
			calculateHistogram(flowMag, flowAngle, rectangle, it->hof, 9, p.t, true);
			calculateHistogram(flowXMag, flowXAngle, rectangle, it->mbhX, 8, p.t, false);
			calculateHistogram(flowYMag, flowYAngle, rectangle, it->mbhY, 8, p.t, false);*/



			if (it->getSize() == maxTrajectoryLength_) {
				it->isMaxLengthAchieved = true;
			}
			else {
				Point new_p(new_x, new_y, p.t+1);
				it->addPoint(new_p);
			}
		}
		++it;
	}
}

bool Program::isValid(Trajectory& traj) {

	f32 meanX = 0, meanY = 0, varX = 0, varY = 0;
	for (u32 i=0; i<traj.getSize(); i++) {
		meanX += traj.track.at(i).x;
		meanY += traj.track.at(i).y;
	}
	meanX /= traj.getSize();
	meanY /= traj.getSize();

	for (u32 i=0; i<traj.getSize(); i++) {
		varX += (traj.track.at(i).x - meanX) * (traj.track.at(i).x - meanX);
		varY += (traj.track.at(i).y - meanY) * (traj.track.at(i).y - meanY);
	}

	varX /= traj.getSize();
	varY /= traj.getSize();
	varX = sqrt(varX);
	varY = sqrt(varY);

	if (varX < minTrajVariance_ || varY < minTrajVariance_) {
		return false;
	}


	///////////////////////
	//Pruning trajectories with sudden large displacements
	f32 length = 0;
	f32 maxJump = 0;
	for (u32 i=0; i<traj.getSize() - 1; i++) {
		Point p = Point::getDifference(traj.track.at(i+1), traj.track.at(i));
		f32 temp = sqrt(p.x * p.x + p.y * p.y);
		if (temp > maxJump) {
			maxJump = temp;
		}
		length += temp;
	}

	if (maxJump >= 0.7 * length) {
		return false;
	}
	///////////////////////////

	return true;
}

void Program::postProcessTrajectories() {
	std::cout<<"Size:"<<trajectories.size()<<std::endl;
	for (std::list<Trajectory>::iterator it = trajectories.begin(); it != trajectories.end(); ) {

		if (it->getSize() < 15) {
			it = trajectories.erase(it);
			continue;
		}
		else if (it->getSize() == 15) {
			if (!isValid(*it)) {
				it = trajectories.erase(it);
				continue;
			}
			it->finalizeDescriptor();
		}

		++it;
	}
	std::cout<<"Size:"<<trajectories.size()<<std::endl;
}

void Program::calculateTrajectories(std::vector<cv::Mat> &video) {
	std::vector<cv::Mat> flows_x;
	std::vector<cv::Mat> flows_y;
	std::vector<Point> points;
	extractDensePoints(video.at(0), points);
	initializeTrajectories(points);

	calculateOpticalFlow(video, flows_x, flows_y);
	applyMedianFilter(flows_x);
	applyMedianFilter(flows_y);

	for (u32 i=1; i<video.size(); i++) {
		trackTrajectories(video.at(i-1) , flows_x[i-1], flows_y[i-1]);
		points.clear();
		getPointsAlongTrajectories(points);
		extractDensePoints(video.at(i), points);
		initializeTrajectories(points);
	}
	postProcessTrajectories();
}

void Program::applyMedianFilter(std::vector<cv::Mat> &flows) {
	for (u32 i=0; i<flows.size(); i++) {
		cv::medianBlur(flows[i], flows[i], medianFilterSize_);
	}
}

void displayFlow(cv::Mat& flowX, cv::Mat& flowY) {
	cv::Mat mag, angle;
	cv::cartToPolar(flowX, flowY, mag, angle, true);

	double max;
	cv::minMaxLoc(mag, 0, &max);
	mag.convertTo(mag, -1, 1.0/max);

	cv::Mat _hsv[3], hsv;
	_hsv[0] = angle;
	_hsv[1] = cv::Mat::ones(angle.size(), angle.type());
	_hsv[2] = mag;

	cv::merge(_hsv, 3, hsv);

	cv::Mat bgr;

	cv::cvtColor(hsv, bgr, CV_HSV2BGR);

	cv::imshow("flow", bgr);
	cv::waitKey(0);
}

void Program::calculateOpticalFlow(std::vector<cv::Mat> &video, std::vector<cv::Mat> &flows_x, std::vector<cv::Mat> &flows_y) {
	flows_x.clear();
	flows_y.clear();
	for (u32 t=0; t < video.size() - 1; t++) {
		cv::Mat flow;
		cv::Mat flowXY[2];

		cv::calcOpticalFlowFarneback(video.at(t), video.at(t+1), flow, 0.702, 5, 10, 2, 7, 1.5, cv::OPTFLOW_FARNEBACK_GAUSSIAN );


		cv::split(flow, flowXY);
		flows_x.push_back(flowXY[0]);
		flows_y.push_back(flowXY[1]);

		//displayFlow(flowXY[0], flowXY[1]);
	}
}

void Program::extractDensePoints(cv::Mat &frame, std::vector<Point> &points) {

	u32 width = frame.cols / samplingWindow_;
	u32 height = frame.rows / samplingWindow_;
	s32 x_max = samplingWindow_ * width;
	s32 y_max = samplingWindow_ * height;

	std::vector<u32> counter(width * height);
	for (u32 i=0; i < points.size(); i++) {
		s32 x = cvFloor(points.at(i).x);
		s32 y = cvFloor(points.at(i).y);

		if ( x >= x_max || y >= y_max)
			continue;

		x /= samplingWindow_;
		y /= samplingWindow_;

		counter.at(y * width + x) += 1;
	}

	points.clear();
	cv::Mat minEigenVals;// = cv::Mat::zeros(frame.size(), CV_32FC1);
	cv::cornerMinEigenVal(frame, minEigenVals, 3);
	f64 maxEigenVal = 0.0;
	cv::minMaxLoc(minEigenVals, NULL, &maxEigenVal);
	f32 threshold = quality_ * maxEigenVal;

	u32 index = 0;
	u32 offset = samplingWindow_ / 2;
	for (u32 i=0; i< height; i++) {
		for (u32 j = 0; j< width; j++, index++) {
			if (counter[index] > 0)
				continue;
			u32 x = j * samplingWindow_ + offset;
			u32 y = i * samplingWindow_ + offset;

			if (minEigenVals.at<f32>(y, x) > threshold) {
				points.push_back(Point(x, y, 0));
			}
		}
	}
}
void Program::trajectoriesToMatrix(Math::Matrix<f32>& result) {
	if (trajectories.size() == 0)
		return;

	result.resize(426, trajectories.size());

	u32 col = 0;
	for(std::list<Trajectory>::iterator it = trajectories.begin(); it != trajectories.end(); ++it) {
		Math::copy(426, it->finalDesc.begin(), 1, result.begin() + col * 426, 1);
		col++;
	}
}

void Program::calculateAndSaveDenseTrajectories() {
	Core::AsciiStream in(videoList_, std::ios::in);
	std::string videoFile;
	std::list<Math::Matrix<f32> > features;
	u32 totalNumberOfVectors = 0;
	while (in.getline(videoFile)) {
		//std::cout<<videoFile<<std::endl;
		std::vector<cv::Mat> video;
		Utils::readVideo(videoFile, video);
		calculateTrajectories(video);

		if (trajectories.size() == 0)
			continue;

		Math::Matrix<f32> feature;
		trajectoriesToMatrix(feature);

		totalNumberOfVectors += feature.nColumns();
		features.push_back(feature);
		trajectories.clear();
	}

	Features::SequenceFeatureWriter writer;
	writer.initialize(totalNumberOfVectors, 426, features.size());
	for (std::list<Math::Matrix<f32> >::iterator it = features.begin(); it != features.end(); ++it) {
		writer.write(*it);
	}
	in.close();
}

void Program::calculateAndSavePCA() {
	Core::AsciiStream in(videoList_, std::ios::in);
	std::string videoFile;

	Math::Matrix<f32> mat;

	Features::FeatureReader reader;
	reader.initialize();
	reader.newEpoch();

	mat.resize(reader.featureDimension() , reader.totalNumberOfFeatures());
	u32 col = 0;
	while (reader.hasFeatures()) {
		Math::Vector<f32> feature = reader.next();
		mat.setColumn(col, feature);
		col++;
	}

	std::cout<<mat.nColumns()<<"::"<<mat.nRows()<<std::endl;
	cv::Mat data = cv::Mat::zeros(mat.nColumns(), mat.nRows(), CV_32FC1);
	Utils::matrixToCVMat(mat, data, true);


	cv::PCA pca(data, cv::Mat(), 0, 64);
	Math::Matrix<f32> mean, eigenVectors, eigenValues;


	mean.resize(pca.mean.rows, pca.mean.cols);
	Utils::cvMatToMatrix(pca.mean, mean);

	eigenValues.resize(pca.eigenvalues.rows, pca.eigenvalues.cols);
	Utils::cvMatToMatrix(pca.eigenvalues, eigenValues);

	eigenVectors.resize(pca.eigenvectors.rows, pca.eigenvectors.cols);
	Utils::cvMatToMatrix(pca.eigenvectors, eigenVectors);

	mean.write(pcaParamPath_ + "pca_mean.txt");
	eigenValues.write(pcaParamPath_ + "pca_eigenValues.txt");
	eigenVectors.write(pcaParamPath_ + "pca_eigenVectors.txt");
}

void Program::applyPCAAndSave() {
	Math::Matrix<f32> mean, eigenValues, eigenVectors;
	mean.read(pcaParamPath_ + "pca_mean.txt");
	eigenValues.read(pcaParamPath_ + "pca_eigenValues.txt");
	eigenVectors.read(pcaParamPath_ + "pca_eigenVectors.txt");

	cv::Mat meanCVMat(mean.nRows(), mean.nColumns(), CV_32FC1);
	cv::Mat eigenValsCVMat(eigenValues.nRows(), eigenValues.nColumns(), CV_32FC1);
	cv::Mat eigenVecsCVMat(eigenVectors.nRows(), eigenVectors.nColumns(), CV_32FC1);

	Utils::matrixToCVMat(mean, meanCVMat);
	Utils::matrixToCVMat(eigenValues, eigenValsCVMat);
	Utils::matrixToCVMat(eigenVectors, eigenVecsCVMat);

	cv::PCA pca;
	pca.mean = meanCVMat; pca.eigenvalues = eigenValsCVMat; pca.eigenvectors = eigenVecsCVMat;

	Features::SequenceFeatureReader reader;
	reader.initialize();
	reader.newEpoch();

	Features::SequenceFeatureWriter writer;
	writer.initialize(reader.totalNumberOfFeatures(), 64, reader.totalNumberOfSequences());

	Math::Matrix<f32> resultSeq;
	while(reader.hasSequences()) {
		const Math::Matrix<f32>& sequence = reader.next();
		cv::Mat cvMat(sequence.nColumns(), sequence.nRows(), CV_32FC1);
		Utils::matrixToCVMat(sequence, cvMat, true);

		cv::Mat result;
		pca.project(cvMat, result);
		cv::transpose(result, result);

		resultSeq.resize(result.rows, result.cols);
		Utils::cvMatToMatrix(result, resultSeq);

		writer.write(resultSeq);
	}
}

void Program::calculateFisherVectors() {

	Features::SequenceFeatureReader reader;
	reader.initialize();
	reader.newEpoch();

	Features::FeatureWriter writer;
	writer.initialize(reader.totalNumberOfSequences(), 2*64*64);

	ActionRecognition::Gmm gmm_;
	gmm_.load();

	Math::Vector<f32> weights;
	gmm_.getWeights(weights);

	Math::Matrix<f32> means;
	gmm_.getMeans(means);

	Math::Matrix<f32> sigmas;
	gmm_.getSigmas(sigmas);

	means.write("/home/ahsan/temp/gmm/means.txt");
	sigmas.write("/home/ahsan/temp/gmm/sigmas.txt");
	weights.write("/home/ahsan/temp/gmm/weights.txt");

	Math::Vector<f32> meanDer, sigmaDer;
	meanDer.resize(reader.featureDimension());
	sigmaDer.resize(reader.featureDimension());

	Math::Vector<f32> fVector;
	fVector.resize(weights.nRows() * reader.featureDimension() * 2);

	while (reader.hasSequences()) {
		const Math::Matrix<f32>& sequence = reader.next();
		fVector.fill(0.0f);
		u32 index = 0;
		//iterates over all gaussians
		for (u32 i=0; i<weights.nRows(); i++) {

			Math::Vector<f32> sigma, sigma_sq;
			sigmas.getColumn(i, sigma);
			sigmas.getColumn(i, sigma_sq);
			sigma_sq.elementwiseMultiplication(sigma_sq);

			Math::Vector<f32> mean;
			means.getColumn(i, mean);

			meanDer.fill(0.0f);
			sigmaDer.fill(0.0f);

			for (u32 j=0; j<sequence.nColumns(); j++) {

				Math::Vector<f32> feature1, feature2;
				feature1.resize(reader.featureDimension());
				feature2.resize(reader.featureDimension());
				sequence.getColumn(j, feature1);
				sequence.getColumn(j, feature2);

				Math::Vector<f32> posterior;
				gmm_.predict(feature1, posterior);

				feature1.add(mean, -1.0f);
				feature1.elementwiseDivision(sigma);
				feature1.scale(posterior.at(i));
				meanDer.add(feature1);

				feature2.add(mean, -1.0f);
				feature2.elementwiseMultiplication(feature2);
				feature2.elementwiseDivision(sigma_sq);
				feature2.addConstantElementwise(-1.0f);
				feature2.scale(posterior.at(i));
				sigmaDer.add(feature2);
			}
			meanDer.scale(1.0f/(sequence.nColumns() * sqrt(weights.at(i))));
			sigmaDer.scale(1.0f/(sequence.nColumns() * sqrt(2 * weights.at(i))));

			fVector.copyBlockFromVector(meanDer, 0, index, meanDer.nRows());
			index += meanDer.nRows();
			fVector.copyBlockFromVector(sigmaDer, 0, index, sigmaDer.nRows());
			index += sigmaDer.nRows();
		}
		writer.write(fVector);

	}
}

void Program::calculateGMMParams() {
	Features::FeatureReader reader;
	reader.initialize();
	reader.newEpoch();

	Math::Matrix<f32> data;
	data.resize(reader.featureDimension() ,reader.totalNumberOfFeatures());

	u32 col = 0;
	while(reader.hasFeatures()) {
		const Math::Vector<f32>& feature = reader.next();
		data.setColumn(col, feature);
		col++;
	}



	ActionRecognition::Gmm gmm_;
	gmm_.train(data);
}

/*void calculateFisherVector() {
	Math::Matrix<f32> mat;
	Math::Vector<f32> mean, sigma;
	for(int i=0; i<mat.nColumns(); i++) {
		Math::Vector<f32> vector;
		mat.getColumn(i, vector);

		vector.add(mean, -1.0f);
		vector.elementwiseMultiplication(vector);
		vector.elementwiseDivision(sigma);

	}
}*/

void Program::run() {

	if (videoList_.empty()) {
		Core::Error::msg("dense-trajectory.video-list must not be empty.") << Core::Error::abort;
	}


	switch(Core::Configuration::config(paramTask_)) {
	case claculateAndSaveTrajectories:
		calculateAndSaveDenseTrajectories();
		break;
	case calculatePCA:
		calculateAndSavePCA();
		break;
	case applyPCA:
		applyPCAAndSave();
		break;
	case calculateGMM:
		calculateGMMParams();
		break;
	case calFisherVectors:
		calculateFisherVectors();
		break;
	}
}


