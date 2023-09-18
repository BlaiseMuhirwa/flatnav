#pragma once

#include <algorithm>
#include <iostream>
#include <stdint.h>
#include <string>
#include <string_view>
#include <toml++/toml.h>
#include <unordered_map>
#include <vector>
#include <yaml-cpp/yaml.h>

struct IndexParameterTemplate {
  std::vector<uint32_t> ef_construction;
  std::vector<uint32_t> ef_search;
  std::vector<uint32_t> max_edges_per_node;
};

struct Dataset {
  std::string name;
  std::string index_parameter_template;
  std::string train_filepath;
  std::string query_filepath;
  std::string ground_truth_filepath;

  uint32_t train_size;
  uint32_t test_size;
  uint32_t neighbors;
  uint32_t dim;
  uint32_t metric_id;
};

struct BenchmarksConfig {
  std::vector<Dataset> datasets;
  std::unordered_map<std::string, IndexParameterTemplate>
      index_parameter_templates;
};

BenchmarksConfig parseBenchmarkConfig(const std::string &filename) {

  std::cout << "[info] parsing benchmark config..." << filename << std::endl;
  YAML::Node config = YAML::LoadFile(filename);

  if (!config["index_parameter_templates"]) {
    throw std::runtime_error("Missing index_parameter_templates");
  }

  if (!config["datasets"]) {
    throw std::runtime_error("Missing benchmark datasets");
  }

  BenchmarksConfig benchmarks_config;

  // Parse index parameter templates
  for (YAML::const_iterator it = config["index_parameter_templates"].begin();
       it != config["index_parameter_templates"].end(); it++) {
    IndexParameterTemplate index_parameter_template;
    std::string template_name = it->first.as<std::string>();
    index_parameter_template.ef_construction =
        it->second["ef_construction"].as<std::vector<uint32_t>>();
    index_parameter_template.ef_search =
        it->second["ef_search"].as<std::vector<uint32_t>>();
    index_parameter_template.max_edges_per_node =
        it->second["max_edges_per_node"].as<std::vector<uint32_t>>();

    benchmarks_config.index_parameter_templates[template_name] =
        index_parameter_template;
  }

  std::cout << "[info] parsing datasets..." << std::endl;

  // Parse datasets
  for (const auto &dataset_config : config["datasets"]) {
    Dataset dataset;
    dataset.name = dataset_config["name"].as<std::string>();
    dataset.index_parameter_template =
        dataset_config["index_parameter_template"].as<std::string>();
    dataset.train_filepath = dataset_config["train_file"].as<std::string>();
    dataset.query_filepath = dataset_config["test_file"].as<std::string>();
    dataset.ground_truth_filepath =
        dataset_config["gtruth_file"].as<std::string>();

    dataset.train_size = dataset_config["train_size"].as<uint32_t>();
    dataset.test_size = dataset_config["test_size"].as<uint32_t>();
    dataset.neighbors = dataset_config["neighbors"].as<uint32_t>();
    dataset.dim = dataset_config["dim"].as<uint32_t>();
    dataset.metric_id = dataset_config["metric_id"].as<uint32_t>();

    benchmarks_config.datasets.push_back(dataset);
  }

  return benchmarks_config;
}
