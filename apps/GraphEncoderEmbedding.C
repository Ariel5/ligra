//
// Created by Ariel Lubonja on 10/13/22.
// This file is the unweighted edge version of Graph Encoder Embedding
// Due to Ligra's implementation, it is not straightforward to join the two into a single file
// This code was created by modifying the PageRank.C file in Ligra
//

#include <vector>
#include "ligra.h"
#include "math.h"
#include <chrono>

void print_to_file(const double* Z, string file_name, const int n, const int k);
size_t getCurrentRSS();
size_t getPeakRSS();

// Ariel This file only works unweighted graph -
// No weighed examples in Ligra, despite how supposedly "easy" it is to extend

// Ariel - PRUpdate(s,d) in paper
template<class vertex>
struct PR_F { // Do this to edges. But aren't edges defn. by their vertices?
    double *z_curr;
    double *z_next; // Ariel - these are already vectors! No need to worry about assigning them

    int *Y; // Supervised labels for each vertex. More fitting as memeber of Vertex class but whatever
    vertex *V;
    const int n;
    string laplacian;

    double *W;
    PR_F(double *_z_curr, double *_z_next, const int _n, int *_Y, double *_W, vertex *_V, string _laplacian)
            : // Constructor. Pass arrays by pointer - easiest way to pass arrays in structs
            z_curr(_z_curr), z_next(_z_next), n(_n), Y(_Y), W(_W), V(_V), laplacian(_laplacian) {}


    // Ariel Which is the source and destination vertices?
        // s seems to be DESTINATION! d - SOURCE. Found from debugging. TODO may change
    inline bool update(uintE d, uintE s) { //update function applies PageRank equation
        // Ariel I believe -1 or negative label means don't know - ignored

        // Always use Atomic update
        updateAtomic(d, s)

//        if (Y[s] >= 0)
//            z_next[Y[s] * n + d] += W[Y[s] * n + s];
//        if (Y[d] >= 0 && s != d) // Asymmetric in GEE.py too. Also, in Ligra s,d are swapped
//            z_next[Y[d] * n + s] += W[Y[d] * n + d];

        return 1;
    }

    // Contention possible when neighbors of node have same class, therefore
    // write to the same cell in z_next
    inline bool updateAtomic(uintE d, uintE s) {
        if (Y[s] >= 0) {
            writeAdd(&z_next[Y[s] * n + d], W[Y[s] * n + s]);
        }
        if (Y[d] >= 0 && s != d) {
            writeAdd(&z_next[Y[d] * n + s], W[Y[d] * n + d]);
        }
        return 1;
    }


    inline bool cond(intT d) { return cond_true(d); }
}; // No condition. Apply to all vertices

// Embedding Matrix is kxN - map each vertex to a label. GEE iterates over edges

// PRLocalCompute(i) in Ligra paper
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


// Run GEE
template<class vertex>
void Compute(graph<vertex> &GA, commandLine P) {
    const int k = P.getOptionLongValue("-nClusters", 3);

    // Embedding semi-supervised labels
    const string Y_LOCATION = P.getOptionValue("-yLocation", "None");
    // For benchmark purposes. to avoid loading Y time. Actually not much faster
//    const int randomY = P.getOptionIntValue("-randomY", 0);
    string laplacian = P.getOptionValue("-Laplacian", "false");
    const string saveEmbedding = P.getOptionValue("-saveEmbedding", "true");

    if (laplacian == "true") {
        cout << "\n\n\nWARNING - Use of -Laplacian flag is not supported for the unweighed version of Graph Encoder Embedding";
        cout << "\nWARNING - Please use the same command but running WeighedGraphEncoderEmbedding.C instead";
        cout << "\nWARNING - This parameter is being ignored, and the Adjacency version of GEE is being run";

        laplacian = "false";
    }

    const long long int n = GA.n;

    double *p_curr1 = newA(double, 1); // Not needed
    double *p_next1 = newA(double, n * k + 1);
    { parallel_for (long i = 0; i < n * k; i++) p_next1[i] = 0; } //0 if unchanged
    p_next1[n * k] = NAN;
    bool *frontier = newA(bool, n); // Frontier should be whole graph's edges
    { parallel_for (long i = 0; i < n; i++) frontier[i] = 1; }

    int *Y = newA(int, n); // TODO maybe set some classes to 1. GEE chooses 2 of 5 vertices in class 1

    if (Y_LOCATION != "None") {
//        timer t; t.start();
//        cout << "Loading specified Y file at " + Y_LOCATION;
        string a;
        std::ifstream infile(Y_LOCATION);
        if (infile.fail()) {
            cout << "\n\nSpecified Y file does not exist or cannot be loaded\n\n";
            exit(-1);
        }
        int i = 0;
        if (infile.is_open()) {
            while (std::getline(infile, a)) {
                Y[i] = std::stoi(a);
                i++;
            }
        }
//        t.stop(); t.reportTotal("Y loading time: ");
    }

// nk: 1*n array, contains the number of observations in each class
// W: encoder marix. W[i,k] = {1/nk if Yi==k, otherwise 0}

// Not doing possibility_detected from GEE.py
// TODO this is wrong - racy write on nonzeroYCount
// TODO write a reducer on nonzeroYCount
    int nk[k]; // correct on Facebook graph
    {
        parallel_for (int i = 0; i < k; i++) {
            int nonzeroYCount = 0;

            {
                for (int j = 0; j < n; j++) {// nk = np.count_nonzero(Y[:,0]==i)
                    if (Y[j] == i)
                        nonzeroYCount++;
                }
            }
            nk[i] = nonzeroYCount;
        }
    }

    vertexSubset Frontier(n, n, frontier);

    double *W = newA(double, n * k + 1);
    { parallel_for (long i = 0; i < n * k; i++) W[i] = 0; }
    W[n * k] = NAN;

    { parallel_for (int i = 0; i < n; i++) { // For i in range(Y.shape[0]) in GEE.py
        int k_i = Y[i]; // TODO LOW Using 1D Y
        if (k_i >= 0)
            W[k_i * n + i] = 1.0 / nk[k_i];
    }}
    // So far, W is good

    edgeMap(GA, Frontier, PR_F<vertex>(p_curr1, p_next1, n, Y, W, GA.V, laplacian), 0, no_output);


// Use this to print output to file to test correctness
    if (saveEmbedding == "true")
        print_to_file(p_next1, "./Z_to_check.csv", n, k);

// Use this to check RAM usage
//    cout << "current Residual Set Size (RAM usage): " << (float) getCurrentRSS() / (1024*1024) << " MB\n\n";
//    cout << "Peak Residual Set Size (RAM usage): " << (float) getPeakRSS() / (1024*1024) << " MB\n\n";

    Frontier.del();
    free(p_curr1);
    free(p_next1);
    free(W);
    free(Y);
}

void print_to_file(const double* Z, string file_name, const int n, const int k) {
    cout << "\n\nSaving Z to " << file_name << "\n";
    std::ofstream outfile(file_name);
    if (outfile.is_open()) {
        int i = 0;
        while (i < n) {
            string row = "";
//            if (i % kn == 0)
            for (int j=0; j<k; j++) {
                row.append(std::to_string(Z[j*n + i]));
                if (j != k-1) {
                    row.append(" "); // CSV-like
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