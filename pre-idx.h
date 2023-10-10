#pragma once
#pragma once
#ifndef NO_MANUAL_VECTORIZATION
#ifdef __SSE__
#define USE_SSE
#ifdef __AVX__
#define USE_AVX
#endif
#endif
#endif

#include "HashBasedBooleanSet.h"
#include "GorderPriorityQueue.h"
#include "ExplicitSet.h"
#include "reordering.h"

#include <vector>
#include <queue> // for std::priority_queue
#include <algorithm> // for std::min
#include <utility> // for std::pair
#include <limits> // for std::numeric_limits<T>::max()
#include "SpaceInterface.h"
#include <fstream>
#include <cstring>

#define NUM_THREADS 1


template <typename dist_t, typename label_t>
class Index
{
public:
	// List of currently-supported graph reordering algorithms. See reordering.h for how to implement new reordering methods
	// dist_t is the type of the underlying metric space (i.e. for uint8_t images, dist_t is uint8_t)
	// label_t is a fixed-size 
	// both dist_t and label_t must be POD types - we do not support custom classes here
	enum class GraphOrder {GORDER, IN_DEG, OUT_DEG, RCM, RCM_2HOP, HUB_SORT, HUB_CLUSTER, DBG, BCORDER};
	enum class ProfileOrder {GORDER, RCM};
	typedef std::pair< dist_t, label_t > dist_label_t;

private:
	typedef unsigned int node_id_t; // internal node numbering scheme
	typedef std::pair< dist_t, node_id_t > dist_node_t;

	struct CompareNodes {
		constexpr bool operator()(dist_node_t const &a, dist_node_t const &b)
									const noexcept {return a.first < b.first;} 
	};

	typedef ExplicitSet VisitedSet;
	typedef std::priority_queue< dist_node_t , std::vector< dist_node_t >, CompareNodes > PriorityQueue;

	char* index_memory;

	size_t M;
	size_t data_size_bytes; // size of one data point (we do not support variable-size data e.g. strings)
	size_t node_size_bytes; // Node consists of: ([data] [M links] [data label]). This layout was selected as 
	// the one with the best cache performance after trying several options 
	size_t max_num_nodes; // determines size of internal pre-allocated memory
	size_t cur_num_nodes;

	DistanceFunction<dist_t> distance; // call-by-pointer distance function
	void* distance_param; // TODO: get rid of this shit. 
	// distance_param just contains "dimensionality." While it's often known at compile-time, it can be unpleasant to 
	// specify e.g. via preprocessor directives. Also poses issues for Python libraries, which only know dimensionality at runtime

	VisitedSet is_visited; // remembers which nodes we've visited, to avoid re-computing distances
	// might be a caching problem - need to profile. I would love to get rid of this in beamSearch.

	char* nodeData(const node_id_t& n){
		char* location = index_memory + n*node_size_bytes;
		return location;
	}

	node_id_t* nodeLinks(const node_id_t& n){
		char* location = index_memory + n*node_size_bytes + data_size_bytes;
		return reinterpret_cast<node_id_t*>(location);
	}

	label_t* nodeLabel(const node_id_t& n){
		char* location = index_memory + n*node_size_bytes + data_size_bytes + M*sizeof(node_id_t);
		return reinterpret_cast<label_t*>(location);
	}

	bool allocateNode(void* data, label_t& label, node_id_t& new_node_id){
		if (cur_num_nodes >= max_num_nodes){return false;}
		new_node_id = cur_num_nodes; 

		std::memcpy(nodeData(cur_num_nodes), data, data_size_bytes);
		*(nodeLabel(cur_num_nodes)) = label;

		node_id_t* links = nodeLinks(cur_num_nodes);
		for (int i = 0; i < M; i++){
			links[i] = cur_num_nodes;
		}

		cur_num_nodes++;
		return true;
	}

	PriorityQueue beamSearch(const void* query, const node_id_t entry_node, const int buffer_size){
		// returns an iterable list of node_id_t's, sorted by distance (ascending)
		PriorityQueue neighbors; // W in the paper
		PriorityQueue candidates; // C in the paper

		is_visited.clear();
		dist_t dist = distance(query, nodeData(entry_node), distance_param);
		dist_t max_dist = dist;

		candidates.emplace(-dist, entry_node);
		neighbors.emplace(dist, entry_node);
		is_visited.insert(entry_node);

		while (!candidates.empty()) {
			// get nearest element from candidates
			dist_node_t d_node = candidates.top();
			if ((-d_node.first) > max_dist){
				break;
			}
			candidates.pop();
			node_id_t* d_node_links = nodeLinks(d_node.second);
			for (int i = 0; i < M; i++){
				if (!is_visited[d_node_links[i]]){ // if we haven't visited the node yet
					is_visited.insert(d_node_links[i]);
					dist = distance(query, nodeData(d_node_links[i]), distance_param);
					// Include the node in the buffer if buffer isn't full or if node is closer than a node already in the buffer
					if (neighbors.size() < buffer_size || dist < max_dist) {
						candidates.emplace(-dist, d_node_links[i]);
						neighbors.emplace(dist, d_node_links[i]);
						if (neighbors.size() > buffer_size){
							neighbors.pop();
						}
						if (!neighbors.empty()){
							max_dist = neighbors.top().first;
						}
					}
				}
			}
		}
		return neighbors;
	}

	// This is from Nick's branch (should probably remove before PR).
	inline std::pair<node_id_t, node_id_t> getNodeBlock(node_id_t node,
														node_id_t max_distance_from_node) {
		// We can't use std::max(0, cur_node - max_distance_from_node) because of overflow with uint32
		node_id_t block_start = node < max_distance_from_node ? 0 : node - max_distance_from_node;
		node_id_t block_end = std::min<node_id_t>(node + max_distance_from_node, cur_num_nodes);

		return {block_start, block_end};
	}


	PriorityQueue greedyBlockFlipFlop(const void* query, const node_id_t entry_node, const int buffer_size, node_id_t max_distance_from_node){
		// Iterates the following two steps, until we cannot make progress:
		// (A) Run greedy descent, until we hit a local minimum (cannot walk along an out-edge toward the query)
		// (B) Compute distances to all points within a block surrounding the local minimum.
		// We can trade computation for accuracy by varying the block size. (B) can be done in parallel.

		PriorityQueue neighbors;
		is_visited.clear();

		std::vector<dist_t> block_distances(2 * max_distance_from_node);
		std::fill(block_distances.begin(), block_distances.end(), std::numeric_limits<dist_t>::max());

		dist_t candidate_dist = distance(query, nodeData(entry_node), distance_param);
		dist_t max_dist = candidate_dist;  // Distance to furthest neighbor in buffer.
		dist_t cur_dist = candidate_dist;  // Dist to current active node in the search.
		node_id_t cur_node = entry_node;
		// std::cout<<"Current node: "<<cur_node<<" ("<<cur_dist<<")"<<std::endl;

		neighbors.emplace(candidate_dist, entry_node);
		is_visited.insert(entry_node);

		bool at_global_min = false;
		while (!at_global_min) {  // Continue to run the greedy-block iteration.
			// (A) Run greedy descent until we cannot make progress anymore.
			bool at_local_min = false;
			// std::cout<<"(A) Current node: "<<cur_node<<" ("<<cur_dist<<")"<<std::endl;
			while (!at_local_min) {
				// Do greedy search to find the local min.
				at_local_min = true;
				node_id_t* cur_node_links = nodeLinks(cur_node);
				for (int i = 0; i < M; i++) {
					if (!is_visited[cur_node_links[i]]){  // if we haven't visited the node yet.
						is_visited.insert(cur_node_links[i]);
						candidate_dist = distance(query, nodeData(cur_node_links[i]), distance_param);
						// std::cout<<"\t Visiting node: "<<cur_node_links[i]<<" ("<<candidate_dist<<")"<<std::endl;
						// Include the node in the buffer of neighbors if the buffer isn't
						// full or if node is closer than a node already in the buffer.
						if (neighbors.size() < buffer_size || candidate_dist < max_dist) {
							neighbors.emplace(candidate_dist, cur_node_links[i]);
							if (neighbors.size() > buffer_size){
								neighbors.pop();
							}
							if (!neighbors.empty()){
								max_dist = neighbors.top().first;
							}
						}
						if (candidate_dist < cur_dist) {
							// If this node is better than the current node.
							at_local_min = false;
							cur_dist = candidate_dist;
							cur_node = cur_node_links[i];
						}
					}
				}
			}

			// std::cout<<"(B) Current node: "<<cur_node<<" ("<<cur_dist<<")"<<std::endl;
			// (B) At this point, we have completed the greedy search and found a local minimum.
			// Next, we search within a block to determine whether there is a nearby node from
			// which we can continue the greedy search.
			at_global_min = true;
			auto [block_start, block_end] = getNodeBlock(cur_node, max_distance_from_node);
			// This is necessary so that we skip nodes that were previously visited (and therefore did 
			// not have their distance calculated) when updating the neighbors.
			std::fill(block_distances.begin(), block_distances.end(), std::numeric_limits<dist_t>::max());

			#pragma omp parallel for default(none) shared(block_start, block_end, query, \
													      block_distances) num_threads(NUM_THREADS)
			for (node_id_t block_node = block_start; block_node < block_end; block_node++) {
				if (is_visited[block_node]){
					continue;
				}
				is_visited.insert(block_node);
				dist_t block_node_dist = distance(query, nodeData(block_node), distance_param);
				node_id_t block_index = block_node - block_start;
				block_distances[block_index] = block_node_dist;
			}

			// Now we update the neighbors and the cur_node.
			// std::cout<<"\tBlock Nodes: ";
			for (node_id_t block_node = block_start; block_node < block_end; block_node++){
				node_id_t block_index = block_node - block_start;
				// std::cout<<"\t"<<block_node<<" ("<<candidate_dist<<")" <<std::endl;
				dist_t candidate_dist = block_distances[block_index];
				// If we didn't visit the node, then block_distances = maximum possible distance value
				// and we will ignore the node (prevents us from double-inserting to the PriorityQueue).
				if (neighbors.size() < buffer_size || candidate_dist < max_dist) {
					neighbors.emplace(candidate_dist, block_node);
					if (neighbors.size() > buffer_size){
						neighbors.pop();
					}
					if (!neighbors.empty()){
						max_dist = neighbors.top().first;
					}
				}
				if (candidate_dist < cur_dist){
					// This node is better than cur_node - we can try to continue greedy descent from here.
					at_global_min = false;
					cur_dist = candidate_dist;
					cur_node = block_node;
				}
			}
		}
		return neighbors;
	}


	PriorityQueue beamBlockFlipFlop(const void* query, const node_id_t entry_node, const int buffer_size, node_id_t max_distance_from_node){
		// Iterates the following two steps, until we cannot make progress:
		// (A) Run beam search with a small beam width, until we hit a local minimum (cannot walk along an out-edge toward the query)
		// (B) Compute distances to all points within a block surrounding the local minimum.
		// We can trade computation for accuracy by varying the block size. (B) can be done in parallel.

		PriorityQueue neighbors;
		is_visited.clear();

		std::vector<dist_t> block_distances(2 * max_distance_from_node);
		std::fill(block_distances.begin(), block_distances.end(), std::numeric_limits<dist_t>::max());

		dist_t candidate_dist = distance(query, nodeData(entry_node), distance_param);
		dist_t max_dist = candidate_dist;  // Distance to furthest neighbor in buffer.
		dist_t cur_dist = candidate_dist;  // Dist to current active node in the search.
		node_id_t cur_node = entry_node;
		// std::cout<<"Current node: "<<cur_node<<" ("<<cur_dist<<")"<<std::endl;

		neighbors.emplace(candidate_dist, entry_node);
		is_visited.insert(entry_node);

		bool at_global_min = false;
		while (!at_global_min) {  // Continue to run the greedy-block iteration.
			// (A) Run greedy descent until we cannot make progress anymore.
			bool at_local_min = false;
			// std::cout<<"(A) Current node: "<<cur_node<<" ("<<cur_dist<<")"<<std::endl;
			while (!at_local_min) {
				// Do greedy search to find the local min.
				at_local_min = true;
				node_id_t* cur_node_links = nodeLinks(cur_node);
				for (int i = 0; i < M; i++) {
					if (!is_visited[cur_node_links[i]]){  // if we haven't visited the node yet.
						is_visited.insert(cur_node_links[i]);
						candidate_dist = distance(query, nodeData(cur_node_links[i]), distance_param);
						// std::cout<<"\t Visiting node: "<<cur_node_links[i]<<" ("<<candidate_dist<<")"<<std::endl;
						// Include the node in the buffer of neighbors if the buffer isn't
						// full or if node is closer than a node already in the buffer.
						if (neighbors.size() < buffer_size || candidate_dist < max_dist) {
							neighbors.emplace(candidate_dist, cur_node_links[i]);
							if (neighbors.size() > buffer_size){
								neighbors.pop();
							}
							if (!neighbors.empty()){
								max_dist = neighbors.top().first;
							}
						}
						if (candidate_dist < cur_dist) {
							// If this node is better than the current node.
							at_local_min = false;
							cur_dist = candidate_dist;
							cur_node = cur_node_links[i];
						}
					}
				}
			}

			// std::cout<<"(B) Current node: "<<cur_node<<" ("<<cur_dist<<")"<<std::endl;
			// (B) At this point, we have completed the greedy search and found a local minimum.
			// Next, we search within a block to determine whether there is a nearby node from
			// which we can continue the greedy search.
			at_global_min = true;
			auto [block_start, block_end] = getNodeBlock(cur_node, max_distance_from_node);
			// This is necessary so that we skip nodes that were previously visited (and therefore did 
			// not have their distance calculated) when updating the neighbors.
			std::fill(block_distances.begin(), block_distances.end(), std::numeric_limits<dist_t>::max());

			#pragma omp parallel for default(none) shared(block_start, block_end, query, \
													      block_distances) num_threads(NUM_THREADS)
			for (node_id_t block_node = block_start; block_node < block_end; block_node++) {
				if (is_visited[block_node]){
					continue;
				}
				is_visited.insert(block_node);
				dist_t block_node_dist = distance(query, nodeData(block_node), distance_param);
				node_id_t block_index = block_node - block_start;
				block_distances[block_index] = block_node_dist;
			}

			// Now we update the neighbors and the cur_node.
			// std::cout<<"\tBlock Nodes: ";
			for (node_id_t block_node = block_start; block_node < block_end; block_node++){
				node_id_t block_index = block_node - block_start;
				// std::cout<<"\t"<<block_node<<" ("<<candidate_dist<<")" <<std::endl;
				dist_t candidate_dist = block_distances[block_index];
				// If we didn't visit the node, then block_distances = maximum possible distance value
				// and we will ignore the node (prevents us from double-inserting to the PriorityQueue).
				if (neighbors.size() < buffer_size || candidate_dist < max_dist) {
					neighbors.emplace(candidate_dist, block_node);
					if (neighbors.size() > buffer_size){
						neighbors.pop();
					}
					if (!neighbors.empty()){
						max_dist = neighbors.top().first;
					}
				}
				if (candidate_dist < cur_dist){
					// This node is better than cur_node - we can try to continue greedy descent from here.
					at_global_min = false;
					cur_dist = candidate_dist;
					cur_node = block_node;
				}
			}
		}
		return neighbors;
	}



	// PriorityQueue greedyBlockFlock(const void* query, const node_id_t entry_node, const int buffer_size, node_id_t max_distance_from_node){
	// 	// The idea here is to perform greedy descent to get to a good initialization.
	// 	// Once we've found a good initialization, we perform greedy descent from each un-visited node in a block surrounding the descent location.
	// 	// (i.e. we spawn a flock of greedy descents in a block around the node). We add the nodes visited by each flock to the neighbors queue.
	// 	// Note that we can early-terminate a descent if we run into a visited node, since that node was visited by some other thread 
	// 	// We cannot, however, guarantee that each descent returns unique nodes because of race conditions.
	// 	PriorityQueue neighbors;

	// 	return neighbors;
	// }


  void reprune(node_id_t node){
    node_id_t* links = nodeLinks(node);
    PriorityQueue neighbors;
    for (int i = 0; i < M; i++){
      if (links[i] != node){
        dist_t dist = distance(nodeData(node), nodeData(links[i]), distance_param);
        neighbors.emplace(dist, links[i]);
        links[i] = node;
      }
    }
    selectNeighbors(neighbors, M);
    int i = 0;
    while(neighbors.size() > 0){
      node_id_t neighbor_node_id = neighbors.top().second;
      links[i] = neighbor_node_id;
      i++;
      if (i > M){i = M;}
      neighbors.pop();
    }
  }


	void selectNeighbors(PriorityQueue& neighbors, const int M){
		// selects neighbors from the PriorityQueue, according to HNSW heuristic
		if (neighbors.size() < M) { return; }

		PriorityQueue candidates; 
		std::vector<dist_node_t> saved_candidates; 
		while(neighbors.size() > 0){
			candidates.emplace(-neighbors.top().first, neighbors.top().second);
			neighbors.pop();
		}

		while(candidates.size() > 0){
			if (saved_candidates.size() >= M){ break; }
			dist_node_t current_pair = candidates.top(); 
			candidates.pop();

			bool should_keep_candidate = true; 
			for (const dist_node_t& second_pair : saved_candidates) {
				dist_t cur_dist = distance(nodeData(second_pair.second), nodeData(current_pair.second), distance_param);
				if (cur_dist < (-current_pair.first) ){
					should_keep_candidate = false; 
					break;
				}
			}
			if (should_keep_candidate) {
				saved_candidates.push_back(current_pair); // could do neighbors.emplace except we have to iterate through saved_candidates, and std::priority_queue doesn't support iteration
			}
		}
		// implement my own priority queue, get rid of vector saved_candidates, add directly to neighborqueue earlier
		for (const dist_node_t& current_pair : saved_candidates){
			neighbors.emplace(-current_pair.first, current_pair.second);
		}
	}

	void connectNeighbors(PriorityQueue& neighbors, node_id_t new_node_id){
		// connects neighbors according to the HSNW heuristic
		node_id_t* new_node_links = nodeLinks(new_node_id);
		int i = 0; // iterates through links for "new_node_id"

		while(neighbors.size() > 0){
			node_id_t neighbor_node_id = neighbors.top().second;
			// add link to the current new node
			new_node_links[i] = neighbor_node_id; 
			// now do the back-connections (a little tricky)
			node_id_t* neighbor_node_links = nodeLinks(neighbor_node_id);
			bool is_inserted = false;
			for (int j = 0; j < M; j++){
				if (neighbor_node_links[j] == neighbor_node_id){
					// if there is a self-loop, replace the self-loop with the desired link
					neighbor_node_links[j] = new_node_id;
					is_inserted = true; 
					break;
				}
			}
			if (!is_inserted){
				// now, we may to replace one of the links. This will disconnect the old neighbor and 
				// create a directed edge, so we have to be very careful. To ensure we respect the 
				// pruning heuristic, we construct a candidate set including the old links AND our new one
				// and then prune this candidate set to get the new neighbors
				dist_t max_dist = distance(nodeData(new_node_id), nodeData(neighbor_node_id), distance_param);
				PriorityQueue candidates; 
				candidates.emplace(max_dist, new_node_id);
				for (int j = 0; j < M; j++){
					if (neighbor_node_links[j] != neighbor_node_id){
						candidates.emplace(
							distance(nodeData(neighbor_node_id),nodeData(neighbor_node_links[j]),distance_param),
							neighbor_node_links[j]
							);
					}
				}
				selectNeighbors(candidates, M);
				// connect the pruned set of candidates, including any self-loops:
				int j = 0; 
				while( candidates.size() > 0){ // candidates
					neighbor_node_links[j] = candidates.top().second;
					candidates.pop();
					j++;
				}
				while( j < M ){ // self-loops (unused links)
					neighbor_node_links[j] = neighbor_node_id;
					j++;
				}
			}
			// loop increments:
			i++;
			if (i >= M){i = M;}
			neighbors.pop();
		}


	}

	node_id_t searchInitialization(const void* query, int n_initializations){
		// select entry_node from a set of random entry point options
		int step_size = cur_num_nodes / n_initializations;
		if (step_size <= 0){ step_size = 1; }

		dist_t min_dist = std::numeric_limits<dist_t>::max();
		node_id_t entry_node = 0;

		for( node_id_t node = 0; node < cur_num_nodes; node += step_size){
			dist_t dist = distance(query, nodeData(node), distance_param);
			if (dist < min_dist){
				min_dist = dist; 
				entry_node = node;
			}
		}
		return entry_node;
	}

	inline void swap(node_id_t a, node_id_t b, void* temp_data, node_id_t* temp_links, label_t* temp_label){
		// stash b in temp
		std::memcpy(temp_data, nodeData(b), data_size_bytes);
		std::memcpy(temp_links, nodeLinks(b), M*sizeof(node_id_t));
		std::memcpy(temp_label, nodeLabel(b), sizeof(label_t));

		// place node at a in b
		std::memcpy(nodeData(b), nodeData(a), data_size_bytes);
		std::memcpy(nodeLinks(b), nodeLinks(a), M*sizeof(node_id_t));
		std::memcpy(nodeLabel(b), nodeLabel(a), sizeof(label_t));

		// put node b in a
		std::memcpy(nodeData(a), temp_data, data_size_bytes);
		std::memcpy(nodeLinks(a), temp_links, M*sizeof(node_id_t));
		std::memcpy(nodeLabel(a), temp_label, sizeof(label_t));

		return; 
	}

	void old_relabel(const std::vector<node_id_t>& P){

		char* new_partition = new char[node_size_bytes * max_num_nodes];
		char* old_partition = index_memory;

		for (node_id_t old_node = 0; old_node < cur_num_nodes; old_node++){
			index_memory = old_partition;
			void* old_node_data_p = nodeData(old_node);
			node_id_t* old_node_links = nodeLinks(old_node);
			label_t* old_node_label = nodeLabel(old_node);

			index_memory = new_partition;
			void* new_node_data_p = nodeData(P[old_node]); 
			std::memcpy(new_node_data_p, old_node_data_p, data_size_bytes);

			node_id_t* new_node_links = nodeLinks(P[old_node]);
			for (int i = 0; i < M; i++){
				new_node_links[i] = P[old_node_links[i]];
			}

			label_t* new_node_label = nodeLabel(P[old_node]);
			*(new_node_label) = *(old_node_label);
		}
		// copy old address into new one
		std::memcpy(old_partition, new_partition, node_size_bytes*max_num_nodes);
		// reassign data_partition to be correct
		index_memory = old_partition;
		// free the temporary partition
		delete[] new_partition;
	}



	void relabel(const std::vector<node_id_t>& P){
		// 1. Rewire all of the node connections
		for (node_id_t n = 0; n < cur_num_nodes; n++){
			node_id_t *links = nodeLinks(n);
			for (int m = 0; m < M; m++){
				links[m] = P[links[m]];
			}
		}

		// 2. Physically re-layout the nodes (in place)
		char* temp_data = new char[data_size_bytes];
		node_id_t* temp_links = new node_id_t[M];
		label_t* temp_label = new label_t;

		is_visited.clear(); // a better name in this context would be "is_relocated" (I just didn't want another VisitedList)

		for (node_id_t n = 0; n < cur_num_nodes; n++){
			if ( !is_visited[n] ) {
				
				node_id_t src = n;
				node_id_t dest = P[src];

				// swap node at src with node at dest
				swap(src, dest, temp_data, temp_links, temp_label);

				// mark src as having been relocated
				is_visited.insert(src);

				// recursively relocate the node from "dest"
				while (!is_visited[dest]){
					// mark node as having been relocated
					is_visited.insert(dest);
					// the value of src remains the same. However, dest needs to change because the node
					// located at src was previously located at dest, and must be relocated to P[dest]
					dest = P[dest];

					// swap node at src with node at dest
					swap(src, dest, temp_data, temp_links, temp_label);
				}
			}
		}

		delete[] temp_data;
		delete[] temp_links;
		delete temp_label;
	}

public:

	// N is the max number of nodes. M is the max number of edges. Space provides info about data size and distance function
	Index(SpaceInterface<dist_t> *space, int _N, int _M): 
		max_num_nodes(_N), cur_num_nodes(0), M(_M), is_visited(_N+1) {

		distance_param = space->get_dist_func_param();
		distance = space->get_dist_func();
		data_size_bytes = space->get_data_size();
		node_size_bytes = space->get_data_size() + sizeof(node_id_t)*M + sizeof(label_t);
		size_t index_memory_size = node_size_bytes*max_num_nodes;
		index_memory = new char[index_memory_size];		
	}

	// TODO: change to use a stream rather than string filename for IO
	Index(SpaceInterface<dist_t> *space, std::string& filename):
    	max_num_nodes(0), cur_num_nodes(0), index_memory(NULL) {
		load(filename, space);
	}

	~Index(){
		delete[] index_memory;
	}

	bool add(void* data, label_t& label, int ef_construction, int n_initializations = 100){
		// initialization must happen before alloc due to a stupid bug where searchInitialization chooses new_node_id as the initialization
		// since new_node_id has distance 0 (but no links), this bug literally skips the search
		node_id_t new_node_id;
		node_id_t entry_node = searchInitialization(data, n_initializations);
		// make space for the new node
		if (!allocateNode(data,label,new_node_id)){return false;}
		// search graph for neighbors of new node, connect to them
		if (new_node_id > 0){
			PriorityQueue neighbors = beamSearch(data, entry_node, ef_construction);
			selectNeighbors(neighbors, M);
			connectNeighbors(neighbors, new_node_id);
		} else {return false;}
		return true;
	}

	std::vector< dist_label_t > search(const void* query, const int K, int ef_search, int n_initializations = 100){
		node_id_t entry_node = searchInitialization(query, n_initializations);
		PriorityQueue neighbors = beamSearch(query, entry_node, ef_search);
		std::vector<dist_label_t> results;
		while(neighbors.size() > K){
			neighbors.pop();
		}
		while( neighbors.size() > 0 ){
			results.emplace_back(neighbors.top().first, *nodeLabel(neighbors.top().second));
			neighbors.pop();
		}
		std::sort( results.begin(), results.end(), [](const dist_label_t& left, const dist_label_t& right)
			{ return left.first < right.first; });
		return results;
	}

	std::vector< dist_label_t > searchGBFF(const void* query, const int K, int max_distance_from_node, int n_initializations = 100){
		node_id_t entry_node = searchInitialization(query, n_initializations);
		PriorityQueue neighbors = greedyBlockFlipFlop(query, entry_node, K, max_distance_from_node);
		std::vector<dist_label_t> results;
		while(neighbors.size() > K){
			neighbors.pop();
		}
		while( neighbors.size() > 0 ){
			results.emplace_back(neighbors.top().first, *nodeLabel(neighbors.top().second));
			neighbors.pop();
		}
		std::sort( results.begin(), results.end(), [](const dist_label_t& left, const dist_label_t& right)
			{ return left.first < right.first; });
		return results;
	}


	void save(const std::string& location){
		std::ofstream out(location, std::ios::binary);

		// TODO: proper binary file format, with magic numbers, versioning, checksum
		// data partition and endian checks. Also save as standard-size integers rather than 
		// int or node_id_t (which can be compiler dependent).
		// currently none of this is safe across machines or even across compilers, probably. Really sorry about that.

		out.write(reinterpret_cast< char *>(&M), sizeof(size_t));
		out.write(reinterpret_cast< char *>(&max_num_nodes), sizeof(size_t));
		out.write(reinterpret_cast< char *>(&cur_num_nodes), sizeof(size_t));
		out.write(reinterpret_cast< char *>(&data_size_bytes), sizeof(size_t));
		out.write(reinterpret_cast< char *>(&node_size_bytes), sizeof(size_t));
		
		// write the index partition
		size_t index_memory_size = node_size_bytes*max_num_nodes;
		out.write(reinterpret_cast< char *>(index_memory), index_memory_size);
		out.close();
	}

	void load(const std::string& location, SpaceInterface<dist_t> *space){

		std::ifstream in(location, std::ios::binary);
		in.read(reinterpret_cast< char *>(&M), sizeof(size_t));
		in.read(reinterpret_cast< char *>(&max_num_nodes), sizeof(size_t));
		in.read(reinterpret_cast< char *>(&cur_num_nodes), sizeof(size_t));
		in.read(reinterpret_cast< char *>(&data_size_bytes), sizeof(size_t));
		in.read(reinterpret_cast< char *>(&node_size_bytes), sizeof(size_t));

		if (index_memory != NULL){
			delete[] index_memory;
			index_memory = NULL;
		}

		size_t index_memory_size = node_size_bytes*max_num_nodes;
		index_memory = new char[index_memory_size];
		in.read(reinterpret_cast< char *>(index_memory), index_memory_size);

		distance_param = space->get_dist_func_param(); 
		distance = space->get_dist_func();
		is_visited = VisitedSet(max_num_nodes+1);
		in.close();
	}

	// I don't like this hack for sparsification but I will tolerate it
	std::vector< std::vector<node_id_t> > graph(){
		std::vector< std::vector<node_id_t> > outdegree_table(cur_num_nodes);
		for (node_id_t node = 0; node < cur_num_nodes; node++){
			node_id_t* links = nodeLinks(node);
			for (int i = 0; i < M; i++){
				if (links[i] != node){
					outdegree_table[node].push_back(links[i]);
				}
			}
		}
		return outdegree_table;
	}
	int size(){
		return cur_num_nodes;
	}

	void flash(std::vector< std::vector<node_id_t> >& outdegree_table){
		if (outdegree_table.size() < cur_num_nodes){
			return;
		}

		for (node_id_t node = 0; node < cur_num_nodes; node++){
			node_id_t* links = nodeLinks(node);
			PriorityQueue neighbors;
			for (node_id_t& neighbor_node : outdegree_table[node]){
				dist_t dist = distance(nodeData(node), nodeData(neighbor_node), distance_param);
				neighbors.emplace(dist, neighbor_node);
			}
			while(neighbors.size() > M){
				neighbors.pop();
			}
			// connect neighbors
			for (int i = 0; i < M; i++){
				links[i] = node;
			}
			int i = 0;
			while (neighbors.size() > 0){
				node_id_t neighbor_node_id = neighbors.top().second;
				links[i] = neighbor_node_id;
				i++;
				if (i >= M){i = M;}
				neighbors.pop();
			}
		}
	}



	void reorder(GraphOrder algorithm){
		std::vector< std::vector<node_id_t> > outdegree_table(cur_num_nodes);
		for (node_id_t node = 0; node < cur_num_nodes; node++){
			node_id_t* links = nodeLinks(node);
			for (int i = 0; i < M; i++){
				if (links[i] != node){
					outdegree_table[node].push_back(links[i]);
				}
			}
		}
		std::vector<node_id_t> P;
		// List of algorithms (so far): GORDER, IN_DEG, OUT_DEG, RCM, HUB_SORT, HUB_CLUSTER, DBG, BCORDER
		switch(algorithm){
			case GraphOrder::GORDER      : P = g_order<node_id_t>(outdegree_table, 5); break;
			case GraphOrder::IN_DEG      : P = indegree_order<node_id_t>(outdegree_table); break;
			case GraphOrder::OUT_DEG     : P = outdegree_order<node_id_t>(outdegree_table); break;
			case GraphOrder::RCM         : P = rcm_order<node_id_t>(outdegree_table); break;
			case GraphOrder::RCM_2HOP    : P = rcm_order_2hop<node_id_t>(outdegree_table); break;
			case GraphOrder::HUB_SORT    : P = hubsort_order<node_id_t>(outdegree_table); break;
			case GraphOrder::HUB_CLUSTER : P = hubcluster_order<node_id_t>(outdegree_table); break;
			case GraphOrder::DBG         : P = dbg_order<node_id_t>(outdegree_table, 8); break;
			case GraphOrder::BCORDER     : P = bc_order<node_id_t>(outdegree_table, 5); break;
		}
		relabel(P);
	}

	void profile_reorder(void* queries, int n_queries,
		int ef_search, ProfileOrder algorithm){
		// construct the weighted graph
		std::vector< std::vector<node_id_t> > outdegree_table(cur_num_nodes);
		std::vector< std::vector<float> > outdegree_weights(cur_num_nodes);
		for (node_id_t node = 0; node < cur_num_nodes; node++){
			node_id_t* links = nodeLinks(node);
			for (int i = 0; i < M; i++){
				if (links[i] != node){
					outdegree_table[node].push_back(links[i]);
					outdegree_weights[node].push_back(1.0);
				}
			}
		}
		for (int i = 0; i < n_queries; i++){
			char* q = (char*)(queries) + i*(data_size_bytes);
			profile_search(q, ef_search, outdegree_table, outdegree_weights);
		}

		std::vector<node_id_t> P;
		switch(algorithm){
			case ProfileOrder::GORDER    : P = weighted_g_order<node_id_t>(outdegree_table, outdegree_weights, 5); break;
			case ProfileOrder::RCM       : P = weighted_rcm_order<node_id_t>(outdegree_table, outdegree_weights); break;
		}

		relabel(P);
	}


	void profile_search(const void* query, int ef_search,
		std::vector< std::vector<node_id_t> > &outdegree_table,
		std::vector< std::vector<float> > &outdegree_weights,
		int n_initializations = 100){
		node_id_t entry_node = searchInitialization(query, n_initializations);
		// this is a pasted-in profiled version of beamSearch
		int buffer_size = ef_search;
		// returns an iterable list of node_id_t's, sorted by distance (ascending)
		PriorityQueue neighbors; // W in the paper
		PriorityQueue candidates; // C in the paper

		is_visited.clear();
		dist_t dist = distance(query, nodeData(entry_node), distance_param);
		dist_t max_dist = dist;

		candidates.emplace(-dist, entry_node);
		neighbors.emplace(dist, entry_node);
		is_visited.insert(entry_node);

		while (!candidates.empty()) {
			// get nearest element from candidates
			dist_node_t d_node = candidates.top();
			if ((-d_node.first) > max_dist){
				break;
			}
			candidates.pop();
			node_id_t* d_node_links = nodeLinks(d_node.second);
			for (int i = 0; i < M; i++){
				if (!is_visited[d_node_links[i]]){ // if we haven't visited the node yet
					is_visited.insert(d_node_links[i]);
					dist = distance(query, nodeData(d_node_links[i]), distance_param);
					// we have done the traversal d_node.second -> d_node_links[i]
					// so we have to increment the corresponding weight
					for (int outdegree_index = 0; outdegree_index < outdegree_table[d_node.second].size(); outdegree_index++){
						if (outdegree_table[d_node.second][outdegree_index] == d_node_links[i]){
							outdegree_weights[d_node.second][outdegree_index] += 1;
						}
					}
					// Include the node in the buffer if buffer isn't full or if node is closer than a node already in the buffer
					if (neighbors.size() < buffer_size || dist < max_dist) {
						candidates.emplace(-dist, d_node_links[i]);
						neighbors.emplace(dist, d_node_links[i]);
						if (neighbors.size() > buffer_size){
							neighbors.pop();
						}
						if (!neighbors.empty()){
							max_dist = neighbors.top().first;
						}
					}
				}
			}
		}
	}


	std::vector< int > location_search(const void* query, const int K, int ef_search, int n_initializations = 100){
		node_id_t entry_node = searchInitialization(query, n_initializations);
		PriorityQueue neighbors = beamSearch(query, entry_node, ef_search);
		std::vector<dist_node_t> results;
		while(neighbors.size() > K){
			neighbors.pop();
		}
		while( neighbors.size() > 0 ){
			results.emplace_back(neighbors.top().first, neighbors.top().second);
			neighbors.pop();
		}
		std::sort( results.begin(), results.end(), [](const dist_node_t& left, const dist_node_t& right)
			{ return left.first < right.first; });
		std::vector<int> out(K);
		for (int i = 0; i < K; i++){
			out[i] = results[i].second;
		}
		return out;
	}

};