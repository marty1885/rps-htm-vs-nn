#pragma once

#include <nupic/algorithms/Cells4.hpp>
#include <nupic/algorithms/TemporalMemory.hpp>
#include <nupic/algorithms/SpatialPooler.hpp>
#include <nupic/algorithms/Anomaly.hpp>

using namespace nupic;
using nupic::algorithms::spatial_pooler::SpatialPooler;
using nupic::algorithms::temporal_memory::TemporalMemory;
using nupic::algorithms::Cells4::Cells4;
using nupic::algorithms::anomaly::Anomaly;

inline std::vector<UInt> sparsify(const xt::xarray<bool>& t)
{
	std::vector<UInt> v;
	v.reserve(t.size()/10);
	for(size_t i=0;i<t.size();i++)
	{
		if(t[i])
			v.push_back(i);
	}
	return v;
}

xt::xarray<float> categroize(int num_category, int len_per_category,const xt::xarray<bool>& in)
{
	xt::xarray<float> res = xt::zeros<float>({num_category});
	assert(res.size()*len_per_category == in.size());
	for(size_t i=0;i<in.size();i++)
		res[i/len_per_category] += (float)in[i];
	res /= len_per_category;
	return res;
}

//Converts container from one to another
template<typename ResType, typename InType>
inline ResType as(const InType& shape)
{
	return ResType(shape.begin(), shape.end());
}

struct HTMLayerBase
{
	HTMLayerBase() = default;
	HTMLayerBase(std::vector<size_t> inDim, std::vector<size_t> outDim)
		: inputDimentions(inDim), outputDimentions(outDim){}
	
	std::vector<size_t> inputDimentions;
	std::vector<size_t> outputDimentions;
	
	size_t inputSize()
	{
		size_t s = 1;
		for(auto v : inputDimentions)
			s *= v;
		return s;
	}
	
	size_t outputSize()
	{
		size_t s = 1;
		for(auto v : outputDimentions)
			s *= v;
		return s;
	}
};

struct SP : HTMLayerBase
{
	SP() = default;
	SP(std::vector<size_t> inDim, std::vector<size_t> outDim)
		: HTMLayerBase(inDim, outDim)
	{
		std::vector<UInt> inSize = as<std::vector<UInt>>(inputDimentions);
		std::vector<UInt> outSize = as<std::vector<UInt>>(outputDimentions);
		sp = SpatialPooler(inSize, outSize);
	}
	
	xt::xarray<bool> compute(const xt::xarray<bool>& t, bool learn)
	{
		std::vector<UInt> in(inputSize());
		std::vector<UInt> out(outputSize());
		for(size_t i=0;i<t.size();i++)
			in[i] = t[i];
		
		sp.compute(in.data(), learn, out.data());
		
		xt::xarray<bool> res = xt::zeros<bool>(outputDimentions);
		for(size_t i=0;i<out.size();i++)
			res[i] = out[i];
		return res;
	}

	SpatialPooler* operator-> ()
	{
		return &sp;
	}

	const SpatialPooler* operator-> () const
	{
		return &sp;
	}
	
	SpatialPooler sp;
};

struct TP : HTMLayerBase
{
	TP() = default;
	TP(std::vector<size_t> inDim, size_t numCol)
		: HTMLayerBase(inDim, inDim), colInTP(numCol)
		, tp(inputSize(), colInTP, 6, 6, 15, .1, .21, 0.23, 1.0, .1, .1, 0.002,
			false, 42, true, false)
	{
	}
	
	xt::xarray<bool> compute(const xt::xarray<bool>& t, bool learn)
	{
		std::vector<Real> in(t.begin(), t.end());
		std::vector<Real> out(t.size()*colInTP);
		tp.compute(in.data(), out.data(), true, learn);
		xt::xarray<bool> res = xt::zeros<bool>(outputDimentions);
		for(size_t i=0;i<out.size()/colInTP;i++)
			res[i] = out[i*colInTP];
		return res;
	}

	Cells4* operator-> ()
	{
		return &tp;
	}

	const Cells4* operator-> () const
	{
		return &tp;
	}
	
	void reset()
	{
		tp.reset();
	}
	
	size_t colInTP;
	Cells4 tp;
};

struct TM : HTMLayerBase
{
	TM() = default;
	TM(std::vector<size_t> inDim, size_t numCol, size_t maxSegmentsPerCell=255, size_t maxSynapsesPerSegment=255)
		: HTMLayerBase(inDim, inDim), colInTP(numCol)
	{
		std::vector<UInt> inSize = as<std::vector<UInt>>(inputDimentions);
		tm = TemporalMemory(inSize, numCol, 13, 0.21, 0.5, 10, 20, 0.1, 0.1, 0, 42, maxSegmentsPerCell, maxSynapsesPerSegment, true);
	}
	
	xt::xarray<bool> compute(const xt::xarray<bool>& t, bool learn)
	{
		std::vector<UInt> cols = sparsify(t);
		xt::xarray<bool> tpOutput = xt::zeros<bool>(t.shape());
		tm.compute(cols.size(), &cols[0], learn);
		auto next = tm.getPredictiveCells();
		for(auto idx : next)
			tpOutput[idx/colInTP] = true;
		return tpOutput;
	}

	TemporalMemory* operator-> ()
	{
		return &tm;
	}

	const TemporalMemory* operator-> () const
	{
		return &tm;
	}
	
	void reset()
	{
		tm.reset();
	}
	
	size_t colInTP;
	TemporalMemory tm;
};

struct Anom
{
	float operator() (xt::xarray<bool> a, xt::xarray<bool> b)
	{
		return anomaly.compute(sparsify(a), sparsify(b));
	}
	Anomaly anomaly;
};
