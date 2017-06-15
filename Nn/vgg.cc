#include "vgg.hh"

const Core::ParameterString vgg::paramVideoList_("video-list", "", "vgg");

vgg::vgg():
    videoList_(Core::Configuration::config(paramVideoList_))
{
}
void vgg::prepareVideoForTemporalStream(){
    if (videoList_.empty())
        Core::Error::msg("vgg.video-list must not be empty.") << Core::Error::abort;

    Core::AsciiStream in(videoList_, std::ios::in);
    std::string videoname;

    while (in.getline(videoname)) {
        createTemporalFramesFromVideo(videoname);
    }
}
void vgg::createTemporalFramesFromVideo(const std::string& filename){
   Video video ;
   readVideo(filename,video);

   for (int i = 0 ; i < video.size()-11;i++){
       cv::Mat c10Image(video[0].rows*20,video[0].cols,CV_8U);
       for (int l = 0 ; l < 10 ; l ++){
           cv::Mat tmpFlow;
           cv::Mat tmpXY[2];
           cv::calcOpticalFlowFarneback(video.at(i+l), video.at(i+l+1), tmpFlow, 0.702, 5, 10, 2, 7, 1.5, cv::OPTFLOW_FARNEBACK_GAUSSIAN );
           cv::split(tmpFlow, tmpXY);

           for (int ii = 0 ; ii < video.at(0).rows;ii++){
               for (int jj = 0 ; jj < video.at(0).cols; jj++){
                   c10Image.at<uchar>((ii*20)+l,jj) = tmpXY[0].at<uchar>(ii,jj);
                   c10Image.at<uchar>((ii*20)+l+1,jj) = tmpXY[1].at<uchar>(ii,jj);
               }
           }
       }
       char buff[3];
       sprintf(buff,"%03d",i);

       imwrite(filename +buff+".jpg",c10Image);
   }
}
void vgg::readVideo(const std::string& filename, Video& result) {
    // open video file
    cv::VideoCapture capture(filename);
    if(!capture.isOpened())
        Core::Error::msg() << "Unable to open Video: " << filename << Core::Error::abort;
    cv::Mat frame, tmp;
    result.clear();
    // read all frames
    u32 nFrames = capture.get(CV_CAP_PROP_FRAME_COUNT);
    while ((nFrames > 0) && (capture.read(frame))) {
        if (frame.channels() == 3)
            cv::cvtColor(frame, tmp, CV_BGR2GRAY);
        else
            tmp = frame;
        result.push_back(cv::Mat());
        tmp.convertTo(result.back(), CV_32FC1, 1.0/255.0);
        nFrames--;
    }
    capture.release();
}
