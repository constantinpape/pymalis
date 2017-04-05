#include <boost/pending/disjoint_sets.hpp>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iterator>
#include <map>
#include <numeric>
#include <queue>
#include <utility>
#include <vector>
#include <iostream>

#include "malis_loss_layer.hpp"

class MalisAffinityGraphCompare {
 private:
	const float * mEdgeWeightArray;
 public:
	explicit MalisAffinityGraphCompare(const float * EdgeWeightArray) {
		mEdgeWeightArray = EdgeWeightArray;
	}
	bool operator()(const int64_t& ind1, const int64_t& ind2) const {
		return (mEdgeWeightArray[ind1] > mEdgeWeightArray[ind2]);
	}
};

// Derived from https://github.com/srinituraga/malis/blob/master/matlab/malis_loss_mex.cpp
void MalisLossLayer::malis(const float* conn_data,
		const int conn_num_dims,
		const int* conn_dims,
		const int* nhood_data,
		const int* nhood_dims,
		const int64_t* seg_data, const bool pos,
		float* dloss_data, float* loss_out,
		float *classerr_out, float *rand_index_out) {

	if ((nhood_dims[1] != (conn_num_dims - 1))
			|| (nhood_dims[0] != conn_dims[0])) {
		std::cerr << "nhood and conn dimensions don't match"
				<< " (" << nhood_dims[1] << " vs. " << (conn_num_dims - 1)
				<< " and " << nhood_dims[0] << " vs. "
				<< conn_dims[conn_num_dims - 1] <<")";
	}

	/* Cache for speed to access neighbors */
	// nVert stores (x * y * z)
	int64_t nVert = 1;
	for (int64_t i = 1; i < conn_num_dims; ++i) {
		nVert *= conn_dims[i];
		// std::cout << i << " nVert: " << nVert << std::endl;
	}

	// prodDims stores x, x*y, x*y*z offsets
	std::vector<int64_t> prodDims(conn_num_dims - 1);
	prodDims[conn_num_dims - 2] = 1;
	for (int64_t i = 1; i < conn_num_dims - 1; ++i) {
		prodDims[conn_num_dims - 2 - i] = prodDims[conn_num_dims - 1 - i]
																			* conn_dims[conn_num_dims - i];
		// std::cout << conn_num_dims - 2 - i << " dims: "
		//	 << prodDims[conn_num_dims - 2 - i] << std::endl;
	}

	/* convert n-d offset vectors into linear array offset scalars */
	// nHood is a vector of size #edges

	std::vector<int32_t> nHood(nhood_dims[0]);
	for (int64_t i = 0; i < nhood_dims[0]; ++i) {
		nHood[i] = 0;
		for (int64_t j = 0; j < nhood_dims[1]; ++j) {
			nHood[i] += (int32_t) nhood_data[j + i * nhood_dims[1]] * prodDims[j];
		}
		// std::cout << i << " nHood: " << nHood[i] << std::endl;
	}

	/* Disjoint sets and sparse overlap vectors */
	std::vector<std::map<int64_t, int64_t> > overlap(nVert);
	std::vector<int64_t> rank(nVert);
	std::vector<int64_t> parent(nVert);
	std::map<int64_t, int64_t> segSizes;
	int64_t nLabeledVert = 0;
	int64_t nPairPos = 0;
	boost::disjoint_sets<int64_t*, int64_t*> dsets(&rank[0], &parent[0]);
	// Loop over all seg data items
	for (int64_t i = 0; i < nVert; ++i) {
		dsets.make_set(i);
		if (0 != seg_data[i]) {
			overlap[i].insert(std::pair<int64_t, int64_t>(seg_data[i], 1));
			++nLabeledVert;
			++segSizes[seg_data[i]];
			nPairPos += (segSizes[seg_data[i]] - 1);
		}
	}

	int64_t nPairTot = (nLabeledVert * (nLabeledVert - 1)) / 2;
	int64_t nPairNeg = nPairTot - nPairPos;
	int64_t nPairNorm;

	if (pos) {
		nPairNorm = nPairPos;
	} else {
		nPairNorm = nPairNeg;
	}

	int64_t edgeCount = 0;
	// Loop over #edges
	for (int64_t d = 0, i = 0; d < conn_dims[0]; ++d) {
		// Loop over Z
		for (int64_t z = 0; z < conn_dims[1]; ++z) {
			// Loop over Y
			for (int64_t y = 0; y < conn_dims[2]; ++y) {
				// Loop over X
				for (int64_t x = 0; x < conn_dims[3]; ++x, ++i) {
					// Out-of-bounds check:
					if (!((z + nhood_data[d * nhood_dims[1] + 0] < 0)
							||(z + nhood_data[d * nhood_dims[1] + 0] >= conn_dims[1])
							||(y + nhood_data[d * nhood_dims[1] + 1] < 0)
							||(y + nhood_data[d * nhood_dims[1] + 1] >= conn_dims[2])
							||(x + nhood_data[d * nhood_dims[1] + 2] < 0)
							||(x + nhood_data[d * nhood_dims[1] + 2] >= conn_dims[3]))) {
						++edgeCount;
					}
				}
			}
		}
	}

	/* Sort all the edges in increasing order of weight */
	std::vector<int64_t> pqueue(edgeCount);
	int64_t j = 0;
	// Loop over #edges
	for (int64_t d = 0, i = 0; d < conn_dims[0]; ++d) {
		// Loop over Z
		for (int64_t z = 0; z < conn_dims[1]; ++z) {
			// Loop over Y
			for (int64_t y = 0; y < conn_dims[2]; ++y) {
				// Loop over X
				for (int64_t x = 0; x < conn_dims[3]; ++x, ++i) {
					// Out-of-bounds check:
					if (!((z + nhood_data[d * nhood_dims[1] + 0] < 0)
							||(z + nhood_data[d * nhood_dims[1] + 0] >= conn_dims[1])
							||(y + nhood_data[d * nhood_dims[1] + 1] < 0)
							||(y + nhood_data[d * nhood_dims[1] + 1] >= conn_dims[2])
							||(x + nhood_data[d * nhood_dims[1] + 2] < 0)
							||(x + nhood_data[d * nhood_dims[1] + 2] >= conn_dims[3]))) {
						pqueue[j++] = i;
					}
				}
			}
		}
	}

	pqueue.resize(j);

	std::sort(pqueue.begin(), pqueue.end(),
			 MalisAffinityGraphCompare(conn_data));

	/* Start MST */
	int64_t minEdge;
	int64_t e, v1, v2;
	int64_t set1, set2;
	int64_t nPair = 0;
	double loss = 0, dl = 0;
	int64_t nPairIncorrect = 0;
	std::map<int64_t, int64_t>::iterator it1, it2;

	/* Start Kruskal's */
	for (int64_t i = 0; i < pqueue.size(); ++i) {
		minEdge = pqueue[i];
		// nVert = x * y * z, minEdge in [0, x * y * z * #edges]

		// e: edge dimension
		e = minEdge / nVert;

		// v1: node at edge beginning
		v1 = minEdge % nVert;

		// v2: neighborhood node at edge e
		v2 = v1 + nHood[e];

		// std::cout << "V1: " << v1 << ", V2: " << v2 << std::endl;

		set1 = dsets.find_set(v1);
		set2 = dsets.find_set(v2);


		if (set1 != set2) {
			dsets.link(set1, set2);

			/* compute the dloss for this MST edge */
			for (it1 = overlap[set1].begin(); it1 != overlap[set1].end(); ++it1) {
				for (it2 = overlap[set2].begin(); it2 != overlap[set2].end(); ++it2) {
					nPair = it1->second * it2->second;

					if (pos && (it1->first == it2->first)) {
						// +ve example pairs
						dl = (float(1.0) - conn_data[minEdge]);
						loss += dl * dl * nPair;
						// Use hinge loss
						dloss_data[minEdge] += dl * nPair;
						if (conn_data[minEdge] <= float(0.5)) {	// an error
							nPairIncorrect += nPair;
						}

					} else if ((!pos) && (it1->first != it2->first)) {
						// -ve example pairs
						dl = (-conn_data[minEdge]);
						loss += dl * dl * nPair;
						// Use hinge loss
						dloss_data[minEdge] += dl * nPair;
						if (conn_data[minEdge] > float(0.5)) {	// an error
							nPairIncorrect += nPair;
						}
					}
				}
			}

			if (nPairNorm > 0) {
				dloss_data[minEdge] /= nPairNorm;
			} else {
				dloss_data[minEdge] = 0;
			}

			if (dsets.find_set(set1) == set2) {
				std::swap(set1, set2);
			}

			for (it2 = overlap[set2].begin();
					it2 != overlap[set2].end(); ++it2) {
				it1 = overlap[set1].find(it2->first);
				if (it1 == overlap[set1].end()) {
					overlap[set1].insert(std::pair<int64_t, int64_t>
						(it2->first, it2->second));
				} else {
					it1->second += it2->second;
				}
			}
			overlap[set2].clear();
		}	// end link
	}	// end while

	/* Return items */
	double classerr, randIndex;
	if (nPairNorm > 0) {
		loss /= nPairNorm;
	} else {
		loss = 0;
	}

	// std::cout << "nPairIncorrect: " << nPairIncorrect << std::endl;
	// std::cout << "nPairNorm: " << nPairNorm << std::endl;

	*loss_out = loss;
	classerr = static_cast<double>(nPairIncorrect)
			/ static_cast<double>(nPairNorm);
	*classerr_out = classerr;
	randIndex = 1.0 - static_cast<double>(nPairIncorrect)
			/ static_cast<double>(nPairNorm);
	*rand_index_out = randIndex;
}

void
MalisLossLayer::evaluate(
		size_t width, size_t height, size_t depth,
		const float* affinity_prob,
		const int64_t* gt_labels,
		float* dloss_neg,
		float* dloss_pos) {

	size_t n = width*height*depth;

	// Set up the neighborhood
	nhood_data_.clear();

	// Dimension primary edges (+Z, +Y, +X) only:
	// 1 edge:		+X					(0,0,1)
	// 2 edges:	 +Y, +X			(0,1,0); (0,0,1)
	// 3 edges:	 +Z, +Y, +X	(1,0,0); (0,1,0); (0,0,1)
	for (int i = 0; i < 3; ++i) {
		nhood_data_.push_back((i + 3) % 3 == 0 ? 1 : 0);
		nhood_data_.push_back((i + 2) % 3 == 0 ? 1 : 0);
		nhood_data_.push_back((i + 1) % 3 == 0 ? 1 : 0);
	}

	nhood_dims_.clear();
	nhood_dims_.push_back(3);
	nhood_dims_.push_back(3);

	// gt affinity
	float* gt_affinity = new float[3*n];
	for (int d = 0; d < 3; d++)
	for (size_t z = 0; z < depth-1; z++)
	for (size_t y = 0; y < width-1; y++)
	for (size_t x = 0; x < height-1; x++) {

		size_t zp = z + nhood_data_[d*3 + 0];
		size_t yp = y + nhood_data_[d*3 + 1];
		size_t xp = x + nhood_data_[d*3 + 2];

		size_t i = z*width*depth + y*width + x;
		size_t j = zp*width*depth + yp*width + xp;

		if (gt_labels[i] == gt_labels[j])
			gt_affinity[d*n + z*width*depth + y*width + x] = 1;
		else
			gt_affinity[d*n + z*width*depth + y*width + x] = 0;
	}

	float* affinity_data_pos = new float[3*n];
	float* affinity_data_neg = new float[3*n];

	for (size_t i = 0; i < n; ++i) {
		affinity_data_pos[i] = std::min(affinity_prob[i], gt_affinity[i]);
		affinity_data_neg[i] = std::max(affinity_prob[i], gt_affinity[i]);
		dloss_pos[i] = 0;
		dloss_neg[i] = 0;
	}

	float loss = 0;
	float loss_out = 0;
	float classerr_out = 0;
	float rand_index_out = 0;

	std::vector<int> aff_dims = {3, (int)depth, (int)height, (int)width};
	malis(
			affinity_data_neg,
			4, &aff_dims[0],
			&nhood_data_[0],
			&nhood_dims_[0],
			gt_labels,
			false, // positive pass?
			dloss_neg,
			&loss_out,
			&classerr_out,
			&rand_index_out);

	loss += 0.5 * loss_out;
	// std::cout << "NEG: " << loss_out << std::endl;

	malis(
			affinity_data_pos,
			4, &aff_dims[0],
			&nhood_data_[0],
			&nhood_dims_[0],
			gt_labels,
			true, // positive pass?
			dloss_pos,
			&loss_out,
			&classerr_out,
			&rand_index_out);

	loss += 0.5 * loss_out;
	// std::cout << "POS: " << loss_out << std::endl;

	delete[] gt_affinity;
	delete[] affinity_data_pos;
	delete[] affinity_data_neg;
}
