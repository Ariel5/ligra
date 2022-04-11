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

// TODO Ariel Assuming unweighted graph - No weighed examples in Ligra, despite how supposedly "easy" it is to extend

// Ariel - PRUpdate(s,d) in paper
template<class vertex>
struct PR_F { // Do this to edges. But aren't edges defn. by their vertices?
    double *z_curr, *z_next; // Ariel - these are already vectors! No need to worry about assigning them
    int *Y; // Supervised labels for each vertex. More fitting as memeber of Vertex class but whatever
    float *W; // Projection matrix

    vertex *V;

    PR_F(double *_z_curr, double *_z_next, int *_Y, float *_W, vertex *_V)
            : // Constructor. Pass arrays by pointer - easiest way to pass arrays in structs
            z_curr(_z_curr), z_next(_z_next), Y(_Y), W(_W), V(_V) {}

    // Ariel - unable to debug what s is. Try to just use it for now
    // TODO Ariel Which is the source and destination vertices?
    inline bool update(uintE s, uintE d) { //update function applies PageRank equation
//        z_next[d] += z_curr[s]/V[s].getOutDegree(); // Ariel Update vertex values. inline ~= static
        if (Y[d] >= 0) { // v_i in GEE.py = s here. v_j = d
            z_next[s, Y[d]] = z_curr[s, Y[d]] + W[d, Y[d]] *
                                                1; // TODO Ariel Assuming unweighted edges! Ligra has weightedEdge class? Else pass as argument to update()
        }
        if (Y[s] >= 0) {
            z_next[d, Y[s]] = z_curr[d, Y[s]] + W[s, Y[s]] * 1;
        }
        return 1;
    }

    // TODO Ariel Hope this isn't used bcs. I didn't change it lol
    inline bool updateAtomic(uintE s, uintE d) { //atomic Update
        writeAdd(&z_next[d], z_curr[s] / V[s].getOutDegree()); // TODO Ariel When to use this vs. Normal
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
    long maxIters = P.getOptionLongValue("-maxiters", 100);

//    long k = P.getOptionLongValue("-nClusters", 5); // TODO Ariel Impl. this later
    int k = 2;

    const intE n = GA.n;

    double *p_curr = newA(double, n);
    { parallel_for (long i = 0; i < n; i++) p_curr[i] = 0; } // Init all in parallel
    double *p_next = newA(double, n);
    { parallel_for (long i = 0; i < n; i++) p_next[i] = 0; } //0 if unchanged
    bool *frontier = newA(bool, n); // Frontier should be whole graph's edges
    { parallel_for (long i = 0; i < n; i++) frontier[i] = 1; }

    int *Y = newA(int, n); // TODO maybe set some classes to 1. GEE chooses 2 of 5 vertices in class 1
    Y[0] = 0;
    Y[1] = 0;
    Y[2] = 0;
    Y[3] = 1;
    Y[4] = 1; // Same as GEE.py 5x5 case

    // Init Y to 2 labels
//        for (int i = 0; i < n; i++) {
//        Y[i] = (i % 2);
//    }
//    k = Y[:,0].max() + 1

//#nk: 1*n array, contains the number of observations in each class
//#W: encoder marix. W[i,k] = {1/nk if Yi==k, otherwise 0}

    std::vector<float> nk(k, 0.0);  //nk = np.zeros((1,k))
//    std::array<float,[n,k]>::fill(const T& value);
//    W = np.zeros((n,k))
    float W[n][k]; // TODO Should I initialize this with 0s?

    // Not doing possibility_detected
    for (int i = 0; i < k; i++) {
        // TODO Ariel Something weird in GEE code. Just port it to C++ for now
        int nonzeroYCount = 0;
        for (int j = 0; j < n; j++) {// nk = np.count_nonzero(Y[:,0]==i)
            if (Y[i] != i)
                nonzeroYCount++;

            nk[i] = nonzeroYCount;
        }
    }

    for (int i = 0; i < n; i++) { // For i in range(Y.shape[0])
        int k_i = Y[i]; // TODO LOW Using 1D Y
        if (k_i >= 0)
            W[i][k_i] = 1 / nk[k_i];
    }

    double Z_curr[k][n]; // Not init-ed to 0
    double Z_next[k][n]; // Not init-ed to 0

//    for (int i=0; i<GA.m; i++) { // Loop over edges. EdgeMap goes here
//        in
//        X:
//    }

    vertexSubset Frontier(n, n, frontier); // TODO TOP What does this do?

    // TODO TOP Each vertex has (kxn) Z-matrix?
    long iter = 0;
    while (iter++ < maxIters) {
        edgeMap(GA, Frontier, PR_F<vertex>(*Z_curr, *Z_next, Y, *W, GA.V), 0, no_output);
        vertexMap(Frontier, PR_Vertex_F(p_curr, p_next, 0.0, n));

        vertexMap(Frontier, PR_Vertex_Reset(p_curr)); // Reset Values
        swap(p_curr, p_next);
    }
//    cout << "Current Embedding values (Z-projection): " << *p_curr;
//    cout << "W: "<<W;

    // Print p_curr
//    for (int i = 0; i < n; i++) {
//        cout << Z[0][i] << " " << Z[1][i] << "\n";
//    }

    Frontier.del();
    free(p_curr);
    free(p_next);
//    free(W);
}