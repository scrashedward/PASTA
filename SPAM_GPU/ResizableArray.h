/*
 
Copyright (c) 2004, Cornell University
All rights reserved.
 
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
 
   - Redistributions of source code must retain the above copyright notice,
       this list of conditions and the following disclaimer.
   - Redistributions in binary form must reproduce the above copyright
       notice, this list of conditions and the following disclaimer in the
       documentation and/or other materials provided with the distribution.
   - Neither the name of Cornell University nor the names of its
       contributors may be used to endorse or promote products derived from
       this software without specific prior written permission.
 
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
 
*/


/////////////////////////////////////////////////////////////////////
///
/// ResizableArray.h
///
/////////////////////////////////////////////////////////////////////
#ifndef __ResizableArray_H__
#define __ResizableArray_H__

/////////////////////////////////////////////////////////////////////
/// @addtogroup GlobalFunctions
/** @{ */

/////////////////////////////////////////////////////////////////////
/// A data structure that provides a resizable array
/// This is used within Spam to avoid reading input data files multiple times,
/// without having to use any STL library classes
/////////////////////////////////////////////////////////////////////
class ResizableArray
{
public:
    /////////////////////////////////////////////////////////////////////
    /// Allocate the memory for the initial size of the array
    ///
    /// @param initialSize
    /////////////////////////////////////////////////////////////////////
    ResizableArray(int initialSize)
    {
        _memory = new int[initialSize];
        memSize = initialSize;
        length = 0;
    };

    /////////////////////////////////////////////////////////////////////
    /// Allocate the memory for the default initial size of the array
    ///
    /////////////////////////////////////////////////////////////////////
    ResizableArray()
    {
        ResizableArray(DEFAULT_INITIAL_SIZE);
    };

    /////////////////////////////////////////////////////////////////////
    /// Destructor
    /////////////////////////////////////////////////////////////////////
    ~ResizableArray()
    {
        delete [] _memory;
    };

    /////////////////////////////////////////////////////////////////////
    /// Push an item onto the end of the array, resizing
    /// the local copy in memory if necessary
    ///
    /// @param item The item to add
    /////////////////////////////////////////////////////////////////////
    void Add(int item)
    {
        if (length < memSize)
        {
            _memory[length] = item;
        }
        else
        {
            // Double the size of our local array

            // Allocate a temp array with double the size
            int * newMem = new int[memSize*2];
            memcpy(newMem, _memory, sizeof(int)*memSize);
            delete [] _memory;

            // Set _memory to temp
            _memory = newMem;
            _memory[length] = item;
            memSize = memSize*2;
        }
        length++;
    };

    /////////////////////////////////////////////////////////////////////
    /// Return a copy of the data in the ResizableArray
    ///
    /// @param arr The memory location to copy the array
    /// @param len The number of elements in the array
    /////////////////////////////////////////////////////////////////////
    void ToArray(int *&arr, int &len)
    {
        len = length;
        arr = new int[len];
        memcpy(arr, _memory, sizeof(int)*len);
    };

private:
    static const int DEFAULT_INITIAL_SIZE = 16;

    // --- private variables
    int * _memory;	// The internal array memory
    int memSize;		// The actual size of memory that is allocated
    int length;			// The number of items the user has added
}
; // class ResizableArray

/** @} */
#endif

