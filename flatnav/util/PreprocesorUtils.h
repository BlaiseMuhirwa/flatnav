#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

namespace flatnav::util {

struct Graph {
  std::vector<std::vector<uint32_t>> adjacency_list;
  int num_vertices;
  int max_num_edges;
};

// Function to load a graph from a Matrix Market file
// TODO: This might need a more robust error handling.
Graph loadGraphFromMatrixMarket(const char *filename) {
  std::ifstream input_file;
  input_file.open(filename);

  if (!input_file.is_open()) {
    std::cerr << "Error opening file" << std::endl;
    exit(1);
  }

  std::string line;
  // Skip the header
  while (std::getline(input_file, line)) {
    if (line[0] != '%')
      break;
  }

  std::istringstream iss(line);
  int num_vertices, num_edges;
  iss >> num_vertices >> num_vertices >> num_edges;

  // Initialize graph
  Graph graph;
  graph.num_vertices = num_vertices;
  graph.max_num_edges = num_edges;
  graph.adjacency_list.resize(num_vertices);

  int u, v;
  while (input_file >> u >> v) {
    // Adjust for 1-based indexing in Matrix Market format
    u--;
    v--;
    graph.adjacency_list[u].push_back(v);
  }

  input_file.close();
  return graph;
}

} // namespace flatnav::util