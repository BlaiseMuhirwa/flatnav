
#include <gSparse/gSparse.hpp>

void sparsifyGraph(const std::string &filename) {
  // Load the graph from MTX file
  gSparse::Graph graph = gSparse::COO::LoadGraphFromMTX(filename);

  // Sparsify the graph
  float epsilon = 0.1;
  gSparse::SpectralSparsifier sparsifier;
  gSparse::Graph sparsified_graph = sparsifier.SpectralSparsify(graph, epsilon);

  std::string output_filename = "sparsified_" + filename;
  gSparse::COO::SaveGraphToMTX(sparsified_graph, output_filename);
}

int main(int argc, char **argv) {

  if (argc < 2) {
    std::clog << "Usage: " << std::endl;
    std::clog << "\t <graph.mtx>: input file" << std::endl;
    return -1;
  }

  std::string filename(argv[1]);

  sparsifyGraph(filename);

  return 1;
}