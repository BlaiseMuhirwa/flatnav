#pragma once

#include <cstdint>
#include <queue>
#include <utility>
#include <vector>

namespace flatnav {

template <typename label_t>
using index_dist_label_t = std::pair<float, label_t>;
using index_node_id_t = uint32_t;
using index_dist_node_t = std::pair<float, index_node_id_t>;

struct index_compare_by_first {
  constexpr bool operator()(index_dist_node_t const& a,
                            index_dist_node_t const& b) const noexcept {
    return a.first < b.first;
  }
};

using index_priority_queue_t =
    std::priority_queue<index_dist_node_t, std::vector<index_dist_node_t>,
                        index_compare_by_first>;

}  // namespace flatnav