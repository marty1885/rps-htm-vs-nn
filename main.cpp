#define CNN_SINGLE_THREAD
#include <cstdlib>
#include <iostream>
#include <vector>

#include "tiny_dnn/tiny_dnn.h"
#include "tiny_dnn/xtensor/xadapt.hpp"
#include "tiny_dnn/xtensor/xio.hpp"

#include "htmhelper.hpp"

//parameters for RNNPlayer
const int RNN_DATA_PER_EPOCH = 3;

using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;

template <typename N>
void constructNet(N &nn, const std::string rnn_type) {
	using recurrent = tiny_dnn::recurrent_layer;
	const int hidden_size = 10; // recurrent state size
	const int seq_len = RNN_DATA_PER_EPOCH; // input sequence length

	if (rnn_type == "rnn") {
		nn << recurrent(rnn(3, hidden_size), seq_len);
	} else if (rnn_type == "gru") {
		nn << recurrent(gru(3, hidden_size), seq_len);
	} else if (rnn_type == "lstm") {
		nn << recurrent(lstm(3, hidden_size), seq_len);
	}
	nn << leaky_relu() << fc(hidden_size, 3) << softmax();
	
	nn.template at<recurrent_layer>(0).bptt_max(RNN_DATA_PER_EPOCH);
}

size_t argmax(xt::xarray<float> a)
{
	float v = a[0];
	size_t idx = 0;
	for(size_t i=1;i<a.size();i++) {
		if(v < a[i]) {
			v = a[i];
			idx = i;
		}
	}
	return idx;
}

struct RNNPlayer
{
public:
	RNNPlayer()
	{
		constructNet(nn_, "gru");
		nn_.set_netphase(net_phase::test);
		nn_.at<recurrent_layer>(0).seq_len(1);
	}
	
	xt::xarray<float> compute(int last_oppo_move)
	{
		xt::xarray<float> in = xt::zeros<float>({3});
		in[last_oppo_move%3] = 1;
		return compute(in);
	}
	
	xt::xarray<float> compute(xt::xarray<float> input)
	{
		assert(input.size() == 3);
		//save data for traning
		if(last_input_.size() != 0) {
			for(auto v : last_input_)
				input_.push_back(v);
			for(auto v : input)
				output_.push_back(v);
		}
		last_input_ = vec_t(input.begin(), input.end());
		
		if(input_.size() == RNN_DATA_PER_EPOCH) {
			assert(input_.size() == output_.size());
			nn_.at<recurrent_layer>(0).seq_len(RNN_DATA_PER_EPOCH);
			nn_.set_netphase(net_phase::train);
			nn_.fit<cross_entropy_multiclass>(optimizer_, std::vector<vec_t>({input_}),std::vector<vec_t>({output_}), 1, 1, [](){},[](){});
			nn_.set_netphase(net_phase::test);
			nn_.at<recurrent_layer>(0).seq_len(1);
			
			input_.clear();
			output_.clear();
		}
		
		
		vec_t out = nn_.predict(vec_t(input.begin(), input.end()));
		
		assert(out.size() == 3);
		xt::xarray<float> r = xt::zeros<float>({3});
		for(size_t i=0;i<out.size();i++)
			r[i] = out[i];
		
		return r;
		
	}
	
	vec_t input_;
	vec_t output_;
	network<sequential> nn_;
	nesterov_momentum optimizer_;
	vec_t last_input_;
};

//parameters for HTMPlayer
const int TP_DEPTH = 32;
const int ENCODE_WIDTH = 24;

xt::xarray<bool> encode(int i)
{
	xt::xarray<bool> res = xt::zeros<bool>({3, ENCODE_WIDTH});
	xt::view(res, i) = true;
	return res;
}

struct HTMPlayer
{
public:
	HTMPlayer() :
		tm_({3, ENCODE_WIDTH}, TP_DEPTH)
	{
		tm_->setMaxNewSynapseCount(64);
		tm_->setPermanenceIncrement(0.1);
		tm_->setPermanenceDecrement(0.045);
		tm_->setConnectedPermanence(0.4);
		tm_->setPredictedSegmentDecrement(0.3*2.0f*tm_->getPermanenceIncrement());
	}
	
	xt::xarray<float> compute(int last_oppo_move, bool learn = true)
	{
		auto out = compute(encode(last_oppo_move), true);
		//std::cout << out << std::endl;
		return categroize(3, ENCODE_WIDTH, out);
	}
	
	xt::xarray<bool> train(const xt::xarray<bool>& x) {return compute(x, true);}
	xt::xarray<bool> predict(const xt::xarray<bool>& x) {return compute(x, false);}
	xt::xarray<bool> compute(const xt::xarray<bool>& x, bool learn) {return tm_.compute(x, learn);}
	void reset() {tm_.reset();}

	TM tm_;
};


enum Move
{
	Rock,
	Paper,
	Scissor
};

int predToMove(int pred)
{
	if(pred == Rock)
		return Paper;
	else if(pred == Paper)
		return Scissor;
	else
		return Rock;
}

int winner(int move1, int move2)
{
	if(move1 == move2)
		return 0;
	if(move1 == Rock && move2 == Paper)
		return -1;
	
	if(move1 == Paper && move2 == Scissor)
		return -1;
	
	if(move1 == Scissor && move2 == Rock)
		return -1;
	
	return 1;
}

std::string move2String(int move)
{
	if(move == Rock)
		return "Rock";
	else if(move == Paper)
		return "Paper";
	return "Scissor";
}

xt::xarray<float> softmax(const xt::xarray<float>& x)
{
	auto b = xt::eval(xt::exp(x-xt::amax(x)[0]));
	return b/xt::sum(b)[0];
}

int main()
{
	//Initialize both AI
	RNNPlayer player1;
	HTMPlayer player2;
	
	int rnn_last_move = 0;
	int htm_last_move = 0;

	size_t rnn_win = 0;
	size_t draw = 0;
	size_t htm_win = 0;

	int num_games = 1000*100;
	for(int i=0;i<num_games;i++) {
		//Run RNN
		auto rnn_out = player1.compute(htm_last_move);
		int rnn_pred = argmax(rnn_out);
		
		//Run HTM
		auto htm_out = player2.compute(rnn_last_move);
		int htm_pred = argmax(htm_out);
		
		int rnn_move = predToMove(rnn_pred);
		int htm_move = predToMove(htm_pred);
		
		int winner_algo = winner(rnn_move, htm_move);
		std::cout << "Round " << i << std::endl;
		//std::cout << "RNN pred: " << rnn_out << ", HTM pred: " << ::softmax(htm_out) << std::endl;
		//std::cout << "RNN: " << move2String(rnn_move) << ", " << "HTM: " << move2String(htm_move)
		//	<< ", Winner: "<< (winner_algo==1?"RNN":(winner_algo==0?"draw":"HTM")) << std::endl;
		//std::cout << std::endl;

		
		rnn_last_move = rnn_move;
		htm_last_move = htm_move;
		
		if(winner_algo == 1)
			rnn_win += 1;
		else if(winner_algo == 0)
			draw += 1;
		else
			htm_win += 1;
	}

	std::cout << "After all the battles" << std::endl;
	std::cout << "RNN Wins " << rnn_win << " times, " << (float)rnn_win/num_games << "%\n";
	std::cout << "HTM Wins " << htm_win << " times, " << (float)htm_win/num_games << "%\n";
	std::cout << "draw: " << draw << std::endl;


}
