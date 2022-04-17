// This code is part of the project "Ligra: A Lightweight Graph Processing
// Framework for Shared Memory", presented at Principles and Practice of
// Parallel Programming, 2013.
// Copyright (c) 2013 Julian Shun and Guy Blelloch
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#include <vector>
#include "ligra.h"
#include "math.h"

void print_to_file(const double* Z, string file_name, const int n, const int k);
size_t getCurrentRSS();
size_t getPeakRSS();

// TODO Ariel Assuming unweighted graph - No weighed examples in Ligra, despite how supposedly "easy" it is to extend

// Ariel - PRUpdate(s,d) in paper
template<class vertex>
struct PR_F { // Do this to edges. But aren't edges defn. by their vertices?
    double *z_curr, *z_next; // Ariel - these are already vectors! No need to worry about assigning them
//    double *z_curr2, *z_next2; // TODO Ariel make matrix later. C++ pointers are fighting me. Now: check correctness
    int *Y; // Supervised labels for each vertex. More fitting as memeber of Vertex class but whatever
    vertex *V;
    const int n;

    // 1st in // 1st in https://stackoverflow.com/questions/8767166/passing-a-2d-array-to-a-c-function
        // "Array initializer must be a list"
//    float W[5][2];
//    PR_F(double *_z_curr1, double *_z_next1,double *_z_curr2, double *_z_next2, int *_Y, float _W[][2], vertex *_V)
//            : // Constructor. Pass arrays by pointer - easiest way to pass arrays in structs
//            z_curr1(_z_curr1), z_next1(_z_next1), z_curr2(_z_curr2), z_next2(_z_next2), Y(_Y), W(_W), V(_V) {}

    // 2nd - "Array initializer must be a list"
//    float *W[2];
//    PR_F(double *_z_curr1, double *_z_next1,double *_z_curr2, double *_z_next2, int *_Y, float *_W[2], vertex *_V)
//            : // Constructor. Pass arrays by pointer - easiest way to pass arrays in structs
//            z_curr1(_z_curr1), z_next1(_z_next1), z_curr2(_z_curr2), z_next2(_z_next2), Y(_Y), W(_W), V(_V) {}

    float *W;
    PR_F(double *_z_curr, double *_z_next, const int _n, int *_Y, float *_W, vertex *_V)
            : // Constructor. Pass arrays by pointer - easiest way to pass arrays in structs
            z_curr(_z_curr), z_next(_z_next), n(_n), Y(_Y), W(_W), V(_V) {}


    // Ariel Which is the source and destination vertices?
        // s seems to be DESTINATION! d - SOURCE. Found from debugging. TODO may change
    inline bool update(uintE s, uintE d) { //update function applies PageRank equation
        // Swap s,d lol
//        uintE temp = d;
//        d = s;
//        s = temp;
//        cout << d << " " << s << "\n";

        // Ariel I believe -1 or negative label means don't know - ignored
        if (Y[d] >= 0) { // TODO Ariel TOP I need some kind of += for curr and next
            z_next[Y[d]*n + s] += W[Y[d]*n + d] * 1; // TODO Ariel Assuming unweighted edges! Ligra has weightedEdge class? Else pass as argument to update()
        }
        if (Y[s] >= 0) {
            z_next[Y[s]*n + d] += W[Y[s]*n + s] * 1;
        }
        return 1;
    }

    // TODO Ariel Hope this isn't used bcs. I didn't change it lol
    inline bool updateAtomic(uintE s, uintE d) { //atomic Update
//        writeAdd(&z_next[d], z_curr[s] / V[s].getOutDegree()); // TODO Ariel When to use this vs. Normal
        return 1;
    }

    inline bool cond(intT d) { return cond_true(d); }
}; // No condition. Apply to all vertices


// TODO Ariel Embedding Matrix is kxN - map each vertex to a label
//  Yet, iterates over edges

// PRLocalCompute(i) in paper
//vertex map function to update its p value according to PageRank equation
struct PR_Vertex_F { // TODO Diff vs. PR_F?
    // TODO Ariel let's reuse p_curr, p_next for self.encoder_embedding
    double damping;
    double addedConstant;
    double *p_curr;
    double *p_next;

    PR_Vertex_F(double *_p_curr, double *_p_next, double _damping, intE n) : // Bigger constructor? Damping?
            p_curr(_p_curr), p_next(_p_next),
            damping(_damping), addedConstant((1 - _damping) * (1 / (double) n)) {}

    inline bool operator()(uintE i) { // TODO Ariel why the index i?
        p_next[i] = damping * p_next[i] + addedConstant;
        return 1;
    }
};

//resets p
struct PR_Vertex_Reset {
    double *p_curr;

    PR_Vertex_Reset(double *_p_curr) :
            p_curr(_p_curr) {}

    inline bool operator()(uintE i) {
        p_curr[i] = 0.0;
        return 1;
    }
};


template<class vertex>
void Compute(graph<vertex> &GA, commandLine P) { // Call PageRank
    const int k = P.getOptionLongValue("-nClusters", 3); // TODO Ariel Impl. this later
    const string graphName = P.getOptionValue("-graphName", "Facebook");
//    int k = 3;

    const intE n = GA.n;
    // Run for nr. of edges
    const long maxIters = P.getOptionLongValue("-maxiters", 1);
    const int divideBy2 = P.getOptionLongValue("-divide", 1);

    double *p_curr1 = newA(double, n*k+1);
    { parallel_for (long i = 0; i < n*k; i++) p_curr1[i] = 0; } // Init all in parallel
    p_curr1[n*k] = NAN;
    double *p_next1 = newA(double, n*k+1);
    { parallel_for (long i = 0; i < n*k; i++) p_next1[i] = 0; } //0 if unchanged
    p_next1[n*k] = NAN;
    bool *frontier = newA(bool, n); // Frontier should be whole graph's edges
    { parallel_for (long i = 0; i < n; i++) frontier[i] = 1; }

    int *Y = newA(int, n); // TODO maybe set some classes to 1. GEE chooses 2 of 5 vertices in class 1

    if (graphName == "Easy") {
        cout << "Easy graph\n";
        { parallel_for (long i = 0; i < n; i++) Y[i] = 0; } // Fill with 0-s
        Y[3] = 1;
        Y[4] = 1; // Same as GEE.py easy 5x5 case
    }
    else if (graphName == "Facebook") {
        cout << "Reading Y-facebook-5percent.txt generated in GEE.py case10 semi-supervised";
        string a;
        std::ifstream infile("../inputs/Y-facebook-5percent.txt");
        if (infile.fail()) {
            cout << "\n\nSpecified Y file does not exist\n\n";
            exit(-1);
        }
        int i = 0;
        if (infile.is_open()) {
            while (std::getline(infile, a)) {
                Y[i] = std::stoi(a);
                i++;
//                if (i == n) { break; }
            }
        }
    }
    else if (graphName == "LiveJournal") {
        cout << "Reading Y-LiveJournal-5percent.txt generated in GEE.py case10 semi-supervised";
        string a;
        std::ifstream infile("../../GraphEmd/Data/liveJournalY.txt");
        int i = 0;
        if (infile.fail()) {
            cout << "\n\nSpecified Y file does not exist\n\n";
            exit(-1);
        }
        if (infile.is_open()) {
            while (std::getline(infile, a)) {
                Y[i] = std::stoi(a);
                i++;
//                if (i == n) { break; }
            }
        }
    }
    else if (graphName == "Twitch") {
        cout << "Reading Twitch Y";
        string a;
        std::ifstream infile("../../../Downloads/twitchFullY-20-removed.txt");
        int i = 0;
        if (infile.fail()) {
            cout << "\n\nSpecified Y file does not exist\n\n";
            exit(-1);
        }
        if (infile.is_open()) {
            while (std::getline(infile, a)) {
                Y[i] = std::stoi(a);
                i++;
            }
        }
    }
    else if (graphName == "Pokec") {
        cout << "Reading Pokec Y";
        string a;
        std::ifstream infile("../../../Downloads/pokec-Y50-sparse.txt");
        int i = 0;
        if (infile.fail()) {
            cout << "\n\nSpecified Y file does not exist\n\n";
            exit(-1);
        }
        if (infile.is_open()) {
            while (std::getline(infile, a)) {
                Y[i] = std::stoi(a);
                i++;
            }
        }
    }
    else {
        cout << "Wrong input graph name. Inputs are case sensitive. Possible inputs: Easy, Facebook, LiveJournal\n\n";
        exit(-1);
    }

//    cout <<
//#nk: 1*n array, contains the number of observations in each class
//#W: encoder marix. W[i,k] = {1/nk if Yi==k, otherwise 0}

    // Not doing possibility_detected
//    int nk[2] = {3,2};
    int nk[k]; // Confirmed correct Facebook graph
    // TODO Ariel implement count_nonzero later. Should return nk = {3,2}
    for (int i = 0; i < k; i++) {
        // TODO Ariel Why need count of indices nk?
        int nonzeroYCount = 0;
        for (int j = 0; j < n; j++) {// nk = np.count_nonzero(Y[:,0]==i)
            if (Y[j] == i)
                nonzeroYCount++;
        }
        nk[i] = nonzeroYCount;
    }

    vertexSubset Frontier(n, n, frontier); // TODO TOP What does this do?

    float *W = newA(float, n*k+1); // W seems ok too, not confirmed tho
    { parallel_for (long i = 0; i < n*k; i++) W[i] = 0; }
    W[n*k] = NAN;

    for (int i = 0; i < n; i++) { // For i in range(Y.shape[0])
        int k_i = Y[i]; // TODO LOW Using 1D Y
        if (k_i >= 0)
            W[k_i*n + i] = 1.0 / nk[k_i];
    }
    // So far, W is good

    // TODO TOP Each vertex has (kxn) Z-matrix?
    long iter = 0;
    while (iter++ < maxIters) {
        edgeMap(GA, Frontier, PR_F<vertex>(p_curr1, p_next1, n, Y, W, GA.V), 0, no_output);

        if (divideBy2 != 0) {
            {
                parallel_for (long i = 0; i < n * k; i++) p_next1[i] /= 2;
            } // TODO lol fix this. Ligra assumes undirected? goes over all edges twice
        }

        cout << "\niter: " << iter << "\n\n";

//        cout << "\n p_next: \t";
//        for (int i = 0; i < n*k; i++) {
//            if (i % n == 0) { cout<<"\n"; }
//            cout << p_next1[i] << "\t";
//        }

//        vertexMap(Frontier, PR_Vertex_Reset(p_curr1)); // Reset Values
//        vertexMap(Frontier, PR_Vertex_Reset(p_curr2));
//        swap(p_curr1, p_next1);
//        swap(p_curr2, p_next2);
    }
//    cout << "Current Embedding values (Z-projection): " << *p_curr;
//    cout << "W: "<<W;

    // Print p_curr
//    for (int i = 0; i < n*k; i++) {
////        if (i % n == 0) { cout<<"\n"; }
//        cout << p_curr1[i] << "\n";
//    }

    cout << "\n\n\n--------------Finished one whole run----------\n\n\n";

//    int debug_placeholder = 5;

    print_to_file(p_next1, "../inputs/Z_output.txt", n, k);
    cout << "current Residual Set Size (RAM usage): " << (float) getCurrentRSS() / (1024*1024) << " MB\n\n";
    cout << "Peak Residual Set Size (RAM usage): " << (float) getPeakRSS() / (1024*1024) << " MB\n\n";

    Frontier.del();
    free(p_curr1);
    free(p_next1);
    free(W);
    free(Y);
}

void print_to_file(const double* Z, string file_name, const int n, const int k) {
    cout << "Saving Z to " << file_name << "\n";
    std::ofstream outfile(file_name);
    if (outfile.is_open()) {
        int i = 0;
        while (i < n) {
            string row = "";
//            if (i % kn == 0)
            for (int j=0; j<k; j++) {
                row.append(std::to_string(Z[j*n + i]));
                if (j != k-1) {
                    row.append(","); // CSV-like
                }
            }
            outfile << row << "\n";
            i++;
        }
    }
}


/* https://stackoverflow.com/questions/669438/how-to-get-memory-usage-at-runtime-using-c
 * Author:  David Robert Nadeau
 * Site:    http://NadeauSoftware.com/
 * License: Creative Commons Attribution 3.0 Unported License
 *          http://creativecommons.org/licenses/by/3.0/deed.en_US
 */

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
#include <unistd.h>
#include <sys/resource.h>

#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
#include <fcntl.h>
#include <procfs.h>

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
#include <stdio.h>

#endif

#else
#error "Cannot define getPeakRSS( ) or getCurrentRSS( ) for an unknown OS."
#endif





/**
 * Returns the peak (maximum so far) resident set size (physical
 * memory use) measured in bytes, or zero if the value cannot be
 * determined on this OS.
 */
size_t getPeakRSS( )
{
#if defined(_WIN32)
    /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo( GetCurrentProcess( ), &info, sizeof(info) );
    return (size_t)info.PeakWorkingSetSize;

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
    /* AIX and Solaris ------------------------------------------ */
    struct psinfo psinfo;
    int fd = -1;
    if ( (fd = open( "/proc/self/psinfo", O_RDONLY )) == -1 )
        return (size_t)0L;      /* Can't open? */
    if ( read( fd, &psinfo, sizeof(psinfo) ) != sizeof(psinfo) )
    {
        close( fd );
        return (size_t)0L;      /* Can't read? */
    }
    close( fd );
    return (size_t)(psinfo.pr_rssize * 1024L);

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
    /* BSD, Linux, and OSX -------------------------------------- */
    struct rusage rusage;
    getrusage( RUSAGE_SELF, &rusage );
#if defined(__APPLE__) && defined(__MACH__)
    return (size_t)rusage.ru_maxrss;
#else
    return (size_t)(rusage.ru_maxrss * 1024L);
#endif

#else
    /* Unknown OS ----------------------------------------------- */
    return (size_t)0L;          /* Unsupported. */
#endif
}





/**
 * Returns the current resident set size (physical memory use) measured
 * in bytes, or zero if the value cannot be determined on this OS.
 */
size_t getCurrentRSS( )
{
#if defined(_WIN32)
    /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo( GetCurrentProcess( ), &info, sizeof(info) );
    return (size_t)info.WorkingSetSize;

#elif defined(__APPLE__) && defined(__MACH__)
    /* OSX ------------------------------------------------------ */
    struct mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if ( task_info( mach_task_self( ), MACH_TASK_BASIC_INFO,
        (task_info_t)&info, &infoCount ) != KERN_SUCCESS )
        return (size_t)0L;      /* Can't access? */
    return (size_t)info.resident_size;

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
    /* Linux ---------------------------------------------------- */
    long rss = 0L;
    FILE* fp = NULL;
    if ( (fp = fopen( "/proc/self/statm", "r" )) == NULL )
        return (size_t)0L;      /* Can't open? */
    if ( fscanf( fp, "%*s%ld", &rss ) != 1 )
    {
        fclose( fp );
        return (size_t)0L;      /* Can't read? */
    }
    fclose( fp );
    return (size_t)rss * (size_t)sysconf( _SC_PAGESIZE);

#else
    /* AIX, BSD, Solaris, and Unknown OS ------------------------ */
    return (size_t)0L;          /* Unsupported. */
#endif
}