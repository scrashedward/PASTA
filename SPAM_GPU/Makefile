OUT=a.out

opt: SPAM.cu
	nvcc -std=c++11 -O3 -arch=sm_50 -o $(OUT) SPAM.cu -D_FORCE_INLINES

nor: SPAM.cu
	nvcc -std=c++11 -arch=sm_50 -o $(OUT) SPAM.cu
