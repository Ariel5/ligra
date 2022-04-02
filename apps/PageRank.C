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
#include "ligra.h"
#include "math.h"

template <class vertex>
struct PR_F {
  double* p_curr, *p_next; // Current and next PR value
  vertex* V;
  PR_F(double* _p_curr, double* _p_next, vertex* _V) : // Constructor?
    p_curr(_p_curr), p_next(_p_next), V(_V) {}
  inline bool update(uintE s, uintE d){ //update function applies PageRank equation
    p_next[d] += p_curr[s]/V[s].getOutDegree(); // Ariel Update vertex values. inline ~= static
    return 1;
  }
  inline bool updateAtomic (uintE s, uintE d) { //atomic Update
    writeAdd(&p_next[d],p_curr[s]/V[s].getOutDegree()); // TODO Ariel When to use this vs. Normal
    return 1;
  }
  inline bool cond (intT d) { return cond_true(d); }}; // No condition. Apply to all vertices

//vertex map function to update its p value according to PageRank equation
struct PR_Vertex_F { // TODO Diff vs. PR_F?
  double damping;
  double addedConstant;
  double* p_curr;
  double* p_next;
  PR_Vertex_F(double* _p_curr, double* _p_next, double _damping, intE n) : // Bigger constructor? Damping?
    p_curr(_p_curr), p_next(_p_next), 
    damping(_damping), addedConstant((1-_damping)*(1/(double)n)){}
  inline bool operator () (uintE i) {
    p_next[i] = damping*p_next[i] + addedConstant;
    return 1;
  }
};

//resets p
struct PR_Vertex_Reset {
  double* p_curr;
  PR_Vertex_Reset(double* _p_curr) :
    p_curr(_p_curr) {}
  inline bool operator () (uintE i) {
    p_curr[i] = 0.0;
    return 1;
  }
};

template <class vertex>
void Compute(graph<vertex>& GA, commandLine P) { // Call PageRank
  long maxIters = P.getOptionLongValue("-maxiters",100);
  const intE n = GA.n;
  const double damping = 0.85, epsilon = 0.0000001;
  
  double one_over_n = 1/(double)n; // Init all to this
  double* p_curr = newA(double,n);
  {parallel_for(long i=0;i<n;i++) p_curr[i] = one_over_n;} // Init all in parallel
  double* p_next = newA(double,n);
  {parallel_for(long i=0;i<n;i++) p_next[i] = 0;} //0 if unchanged
  bool* frontier = newA(bool,n); // TODO ??
  {parallel_for(long i=0;i<n;i++) frontier[i] = 1;} // TOdo All vertices included?

  vertexSubset Frontier(n,n,frontier);
  
  long iter = 0;
  while(iter++ < maxIters) {
    edgeMap(GA,Frontier,PR_F<vertex>(p_curr,p_next,GA.V),0, no_output);
    vertexMap(Frontier,PR_Vertex_F(p_curr,p_next,damping,n));
    //compute L1-norm between p_curr and p_next
    // TODO Ariel Shouldn't this be in LocalCompute? It is in paper
    {parallel_for(long i=0;i<n;i++) {
      p_curr[i] = fabs(p_curr[i]-p_next[i]); // fabs == fast abs?
      }}
    double L1_norm = sequence::plusReduce(p_curr,n);
    if(L1_norm < epsilon) break;
    //reset p_curr
    vertexMap(Frontier,PR_Vertex_Reset(p_curr));
    swap(p_curr,p_next);
  }
  Frontier.del(); free(p_curr); free(p_next); 
}
