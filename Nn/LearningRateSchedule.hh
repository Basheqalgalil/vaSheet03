/*
 * LearningRateSchedule.hh
 *
 *  Created on: Jun 4, 2014
 *      Author: richard
 */

#ifndef NN_LEARNINGRATESCHEDULE_HH_
#define NN_LEARNINGRATESCHEDULE_HH_

#include <Core/CommonHeaders.hh>
#include "Types.hh"
#include "Statistics.hh"

namespace Nn {

/*
 * base class for learning rate schedules
 */
class LearningRateSchedule
{
private:
	static const Core::ParameterEnum paramLearningRateSchedule_;
	static const Core::ParameterFloat paramInitialLearningRate_;
	enum LearningRateScheduleType { none, onlineNewbob, step };
protected:
	Float learningRate_;
	TrainingTask task_;
public:
	LearningRateSchedule();
	virtual ~LearningRateSchedule() {}
	virtual void initialize(TrainingTask task);

	// return new learning rate
	// depending on the schedule, not all parameters may be relevant
	// however, a common interface is necessary
	virtual void updateLearningRate(Statistics& statistics, u32 epoch) {}
	virtual Float learningRate() { return learningRate_; }
	virtual void finalize() {}

	/* factory */
	static LearningRateSchedule* createLearningRateSchedule();
};

/*
 * online newbob learning rate schedule
 */
class OnlineNewbob : public LearningRateSchedule
{
private:
	typedef LearningRateSchedule Precursor;
	static const Core::ParameterFloat paramLearningRateReductionFactor_;
	void logAvgStatistics(u32 epoch);
private:
	Float factor_;
	u32 nClassificationErrors_;
	u32 nObservations_;
	Float oldObjectiveFunction_;
	Float objectiveFunction_;
	u32 oldEpoch_;
	Float error_;
	bool isFirstEpoch_;
public:
	OnlineNewbob();
	virtual ~OnlineNewbob() {}
	virtual void updateLearningRate(Statistics& statistics, u32 epoch);
	virtual void finalize();
};

/*
 * Step learning rate schedule
 */
class StepSchedule : public LearningRateSchedule
{
private:
	typedef LearningRateSchedule Precursor;
	static const Core::ParameterInt paramReduceAfterIterations_;
	static const Core::ParameterFloat paramReductionFactor_;
protected:
	u32 reduceAfterIterations_;
	Float reductionFactor_;
	u32 nIterations_;
public:
	StepSchedule();
	virtual ~StepSchedule() {}
	virtual void initialize(TrainingTask task);
	virtual void updateLearningRate(Statistics& statistics, u32 epoch);
};

} // namespace

#endif /* NN_LEARNINGRATESCHEDULE_HH_ */
