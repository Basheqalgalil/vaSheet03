#include "mybatchnormalizationlayer.hh"

using namespace Nn;

MyBatchNormalizationLayer::MyBatchNormalizationLayer(const char *name):
    Precursor(name)
{

}
void MyBatchNormalizationLayer::initializeParam(Vector &vector, const std::string &basePath, const std::string &suffix, const std::string &paramName, Layer::ParamInitialization initMethod)
{
    struct stat buffer;
    if (suffix.compare("") != 0 &&
            stat(getParamFileName(basePath, paramName, suffix, 0).c_str(), &buffer) == 0) {
        std::string filename = getParamFileName(basePath, paramName, suffix, 0);
        Core::Log::os("Layer ") << name_ << ":" << ": read "<<paramName<<" from " << filename;
        vector.read(filename);
    }
    else if (stat(getParamFileName(basePath, paramName, "", 0).c_str(), &buffer) == 0) {
        std::string filename = getParamFileName(basePath, paramName, "", 0);
        Core::Log::os("Layer ") << name_ << ":" << ": read "<<paramName<<" from " << filename;
        vector.read(filename);
    }
    else {
        _initializeParam(vector, initMethod);
    }
}

void MyBatchNormalizationLayer::initializeParams(const std::string& basePath, const std::string& suffix) {
    require(!isComputing_);

    bias_.resize(1);
    bias_.at(0).resize(2 * nChannels_);

    gamma_.resize(nChannels_);
    beta_.resize(nChannels_);
    gammaDer_.resize(nChannels_ );
    betaDer_.resize(nChannels_ );
    runningMean_.resize(nChannels_);
    runningVariance_.resize(nChannels_);
    saveMean_.resize(nChannels_);
    saveVariance_.resize(nChannels_);

    gammaDer_.setToZero();
    betaDer_.setToZero();
    saveMean_.setToZero();
    saveVariance_.setToZero();

    initializeParam(gamma_, basePath, suffix, "gamma", random);
    initializeParam(beta_, basePath, suffix, "beta", random);
    initializeParam(runningMean_, basePath, suffix, "running-mean", zero);
    initializeParam(runningVariance_, basePath, suffix, "running-variance", zero);
}
void MyBatchNormalizationLayer::save(Vector &vector, const std::string& basePath,
        const std::string& suffix, const std::string& paramName) {
    std::string fn = getParamFileName(basePath, paramName, suffix, 0);
    bool isBiasComputing = vector.isComputing();
    vector.finishComputation();
    vector.write(fn);
    if (isBiasComputing)
        vector.initComputation(false);
}

void MyBatchNormalizationLayer::saveParams(const std::string& basePath, const std::string& suffix) {
    Precursor::saveParams(basePath, suffix);
    gamma_.copyBlockFromVector(bias_.at(0), 0, 0, nChannels_);
    beta_.copyBlockFromVector(bias_.at(0), nChannels_, 0, nChannels_);

    save(beta_, basePath, suffix, "beta");
    save(gamma_, basePath, suffix, "gamma");
    save(runningMean_, basePath, suffix, "running-mean");
    save(runningVariance_, basePath, suffix, "running-variance");
}

void MyBatchNormalizationLayer::forward(u32 port)
{
    Precursor::forward(port);
    u32 t = nTimeframes() - 1;

}
