#pragma once

#include "verifysimd.h"

#include <cstring>
#include <iostream>
// #include <stdint.h> // TODO: Use proper uint32_t everywhere

namespace flatnav {

class ExplicitSet {
  private:
    unsigned short _mark;
    unsigned short *_table;
    unsigned int _tableSize;

  public: 
    ExplicitSet(): _table(NULL), _tableSize(0), _mark(0) {}

    ExplicitSet(const unsigned int size): _table(NULL), _tableSize(0), _mark(0) {
      _mark = 0;
      _tableSize = size;
      _table = new unsigned short[_tableSize]();
    }

    inline void prefetch(const unsigned int num) const {
        #ifdef USE_SSE
            _mm_prefetch((char*)_table[num], _MM_HINT_T0);
        #endif
    }

    inline void insert(const unsigned int num){
      set(num);
    }

    inline void set(const unsigned int num){
      _table[num] = _mark;
    }
    
    inline void reset(const unsigned int num){
      _table[num] = _mark + 1;
    }

    inline void clear(){
      _mark++;
    }

    inline bool operator[](const unsigned int num){
      return (_table[num] == _mark);
    }

    ~ExplicitSet(){
      delete[] _table;
    }

    ExplicitSet(const ExplicitSet& other){ // copy constructor
      _tableSize = other._tableSize;
      _mark = other._mark;
      delete[] _table;
      _table = new unsigned short[_tableSize];
      std::memcpy(other._table, _table, _tableSize * sizeof(unsigned short));
    }

    ExplicitSet(ExplicitSet&& other) noexcept { // move constructor
      _tableSize = other._tableSize;
      _mark = other._mark;
      _table = other._table;
      other._table = NULL;
      other._tableSize = 0;
      other._mark = 0;
      }
  
      ExplicitSet& operator=(const ExplicitSet& other) // copy assignment
      {
          return *this = ExplicitSet(other);
      }
  
      ExplicitSet& operator=(ExplicitSet&& other) noexcept // move assignment
      {
        _tableSize = other._tableSize;
        _mark = other._mark;
        _table = other._table;
        other._table = NULL;
        other._tableSize = 0;
        other._mark = 0;
        return *this;
      }

};

} // namespace flatnav