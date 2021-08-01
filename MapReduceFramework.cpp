#include <iostream>
#include <atomic>
#include <algorithm>
#include "MapReduceFramework.h"
#include "Barrier.h"

using namespace std;

#define SYS_ERR_PREFIX "system error: "

typedef vector<IntermediateVec *> ShuffledVectors;
typedef struct JobContext JobContext;
typedef atomic_ullong State;


/*
 * Description: A thread with its intermediate data.
*/
typedef struct ThreadContext
{
    int ourTid;
    JobContext *jc;
    IntermediateVec *interVec;
    pthread_t thread;

    ~ThreadContext()
    {
        delete interVec;
    }
} ThreadContext;


/*
 * Description: A job description, with input data, and description of the intermediate and output results.
*/
struct JobContext
{
    int numThreads;
    bool waitCalled;
    State state;
    ThreadContext *threadContexts;
    ShuffledVectors *shuffledVectors;
    const InputVec *inputVec;
    OutputVec *outputVec;
    const MapReduceClient *client;
    Barrier *barrier;
    pthread_mutex_t *outputVecMutex;
    atomic_uint mapOldVal;
    atomic_uint shuffledVectorsVal;

    ~JobContext()
    {
        for (auto vec: *shuffledVectors)
        {
            delete vec;
        }
        delete shuffledVectors;
        delete barrier;
        if (pthread_mutex_destroy(outputVecMutex))
        {
            cerr << SYS_ERR_PREFIX << "mutex failed to destroy\n" << endl;
            exit(1);
        }
        delete outputVecMutex;
        delete[] threadContexts;
    }
};


/*
 * Description: The func for sorting the intermediate result.
*/
bool interComp(IntermediatePair &a, IntermediatePair &b)
{
    return *a.first < *b.first;
}


/*
 * Description: The func each thread runs.
*/
void *threadFunc(void *arg)
{
    auto tc = (ThreadContext *) arg;
    tc->jc->state |= (1ULL << 62);
    // Start map stage
    while (true)
    {
        unsigned int idx = tc->jc->mapOldVal++;
        if (idx >= tc->jc->inputVec->size())
        {
            break;
        }
        auto newPair = (*tc->jc->inputVec)[idx];
        tc->jc->client->map(newPair.first, newPair.second, tc);
        tc->jc->state++;
    }
    // End map stage
    sort(tc->interVec->begin(), tc->interVec->end(), interComp);
    tc->jc->barrier->barrier();
    // Start shuffle stage
    if (tc->ourTid == 0)
    {
        unsigned long long numItems = 1ULL << 32;
        for (int i = 0; i < tc->jc->numThreads; ++i)
        {
            numItems += tc->jc->threadContexts[i].interVec->size();
        }
        numItems <<= 31;
        tc->jc->state = numItems;
        while (true)
        {
            K2 *curKey = nullptr;
            for (int i = 0; i < tc->jc->numThreads; ++i)
            {
                auto curInterVec = tc->jc->threadContexts[i].interVec;
                if (!curInterVec->empty())
                {
                    auto candKey = curInterVec->back().first;
                    if (curKey == nullptr || *curKey < *candKey)
                    {
                        curKey = candKey;
                    }
                }
            }
            if (curKey == nullptr)
            {
                break;
            }
            auto newVec = new IntermediateVec();
            for (int i = 0; i < tc->jc->numThreads; ++i)
            {
                auto curInterVec = tc->jc->threadContexts[i].interVec;
                if (!curInterVec->empty())
                {
                    auto candPair = curInterVec->back();
                    if (*candPair.first < *curKey)
                    {
                        continue;
                    }
                    curInterVec->pop_back();
                    newVec->push_back(candPair);
                    tc->jc->state++;
                    --i;
                }
            }
            tc->jc->shuffledVectors->push_back(newVec);
        }
        tc->jc->state = (3ULL << 62) + ((tc->jc->shuffledVectors->size() << 31));
    }
    // End shuffle stage
    tc->jc->barrier->barrier();
    // Start reduce stage
    while (true)
    {
        unsigned int idx = tc->jc->shuffledVectorsVal++;
        if (idx >= tc->jc->shuffledVectors->size())
        {
            break;
        }
        auto curVec = (*tc->jc->shuffledVectors)[idx];
        tc->jc->client->reduce(curVec, tc->jc);
        tc->jc->state++;
    }
    return nullptr;
}


/*
 * Description: Outputs the intermediate results to the intermediate vector of the current thread.
*/
void emit2(K2 *key, V2 *value, void *context)
{
    auto tc = (ThreadContext *) context;
    tc->interVec->push_back(IntermediatePair(key, value));
}

/*
 * Description: Outputs the reduce result to the output vector.
*/
void emit3(K3 *key, V3 *value, void *context)
{
    auto jc = (JobContext *) context;
    pthread_mutex_lock(jc->outputVecMutex);
    jc->outputVec->push_back(OutputPair(key, value));
    pthread_mutex_unlock(jc->outputVecMutex);
}


JobHandle startMapReduceJob(const MapReduceClient &client,
                            const InputVec &inputVec, OutputVec &outputVec,
                            int multiThreadLevel)
{
    auto *jc = new JobContext();
    jc->numThreads = multiThreadLevel;
    jc->waitCalled = false;
    jc->threadContexts = new ThreadContext[multiThreadLevel]; //free all
    jc->shuffledVectors = new ShuffledVectors();
    jc->inputVec = &inputVec;
    jc->outputVec = &outputVec;
    jc->client = &client;
    jc->barrier = new Barrier(multiThreadLevel);
    jc->mapOldVal = 0;
    jc->shuffledVectorsVal = 0;
    unsigned long long pairs = inputVec.size();
    jc->state = pairs << 31;
    jc->outputVecMutex = new pthread_mutex_t;
    if (pthread_mutex_init(jc->outputVecMutex, nullptr))
    {
        cerr << SYS_ERR_PREFIX << "mutex failed to initialize\n" << endl;
        exit(1);
    }
    for (int i = 0; i < multiThreadLevel; ++i)
    {
        auto &curThread = jc->threadContexts[i];
        curThread.ourTid = i;
        curThread.jc = jc;
        curThread.interVec = new IntermediateVec();
        if (pthread_create(&(jc->threadContexts[i].thread), nullptr, threadFunc, (jc->threadContexts) + i))
        {
            cerr << SYS_ERR_PREFIX << "pthread failed to create\n" << endl;
            exit(1);
        }
    }
    return jc;
}


/*
 * Description: Waits for job to finish before starting any new actions in this process.
*/
void waitForJob(JobHandle job)
{
    auto jc = (JobContext *) job;
    if (!jc->waitCalled)
    {
        jc->waitCalled = true;
        for (int i = 0; i < jc->numThreads; ++i)
        {
            pthread_join(jc->threadContexts[i].thread, nullptr);
        }
    }
}


/*
 * Description: Outputs the current stage and the of the job and the complete percentage of it to state.
*/
void getJobState(JobHandle job, JobState *state)
{
    auto jc = (JobContext *) job;
    unsigned long long jState = jc->state.load();
    state->percentage = (float) ((jState << 33) >> 33) / ((jState << 2) >> 33) * 100;
    if (jState >> 62 == 0)
    {
        state->stage = UNDEFINED_STAGE;
    }
    if (jState >> 62 == 1)
    {
        state->stage = MAP_STAGE;
    }
    if (jState >> 62 == 2)
    {
        state->stage = SHUFFLE_STAGE;
    }
    if (jState >> 62 == 3)
    {
        state->stage = REDUCE_STAGE;
    }
}


void closeJobHandle(JobHandle job)
{
    waitForJob(job);
    auto *jc = (JobContext *) job;
    delete jc;
}