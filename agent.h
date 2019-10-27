#pragma once
#include <string>
#include <random>
#include <sstream>
#include <map>
#include <type_traits>
#include <algorithm>
#include "board.h"
#include "action.h"
#include "weight.h"
#include <fstream>

const int num_per_tuple = 16;
const int tuple_num = 4;
const long long tile_per_tuple = 16*16*16*16*16*16*16;

int act =0;
// const std::array<std::array<int, num_per_tuple> ,tuple_num> tuple_feature = {{
// 		{{0,4,8,1,5,9}},

// 		{{1,5,9,2,6,10}},
		
// 		{{2,6,10,14,13,12}},

// 		{{3,7,11,15,14,13}}
// 	}};
const std::array<std::array<int, num_per_tuple> ,tuple_num> tuple_feature = {{
  	{{1, 4, 0, 0,
  	 2, 5, 0, 0,
  	 3, 6, 0, 0,
  	 0, 0, 0, 0}},

  	{{0, 1, 4, 0,
  	 0, 2, 5, 0,
  	 0, 3, 6, 0,
  	 0, 0, 0, 0}},
  	
  	{{0, 0, 1, 0,
  	 0, 0, 2, 0,
  	 0, 0, 3, 0,
  	 0, 0, 4, 0}},

  	{{0, 0, 0, 1,
  	 0, 0, 0, 2,
  	 0, 0, 0, 3,
  	 0, 0, 0, 4}}

}};
const int rt[16] = {3,7,11,15,2,6,10,14,1,5,9,13,0,4,8,12};
const int rf[16] = {3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12};
//the location index of the n-tuple

class agent {
public:
	agent(const std::string& args = "") {
		std::stringstream ss("name=unknown role=unknown " + args);
		for (std::string pair; ss >> pair; ) {
			std::string key = pair.substr(0, pair.find('='));
			std::string value = pair.substr(pair.find('=') + 1);
			meta[key] = { value };
		}
	}
	virtual ~agent() {}
	virtual void open_episode(const std::string& flag = "") {}
	virtual void close_episode(const std::string& flag = "") {}
	virtual action take_action(const board& b) { return action(); }
	virtual bool check_for_win(const board& b) { return false; }

public:
	virtual std::string property(const std::string& key) const { return meta.at(key); }
	virtual void notify(const std::string& msg) { meta[msg.substr(0, msg.find('='))] = { msg.substr(msg.find('=') + 1) }; }
	virtual std::string name() const { return property("name"); }
	virtual std::string role() const { return property("role"); }

protected:
	typedef std::string key;
	struct value {
		std::string value;
		operator std::string() const { return value; }
		template<typename numeric, typename = typename std::enable_if<std::is_arithmetic<numeric>::value, numeric>::type>
		operator numeric() const { return numeric(std::stod(value)); }
	};
	std::map<key, value> meta;
};

class random_agent : public agent {
public:
	random_agent(const std::string& args = "") : agent(args) {
		if (meta.find("seed") != meta.end())
			engine.seed(int(meta["seed"]));
	}
	virtual ~random_agent() {}

protected:
	std::default_random_engine engine;
};

/**
 * base agent for agents with weight tables
 */
class weight_agent : public agent {
public:
	weight_agent(const std::string& args = "") : agent(args) {
		if (meta.find("init") != meta.end()) // pass init=... to initialize the weight
			init_weights(meta["init"]);
		if (meta.find("load") != meta.end()) // pass load=... to load from a specific file
			load_weights(meta["load"]);
	}
	virtual ~weight_agent() {
		if (meta.find("save") != meta.end()) // pass save=... to save to a specific file
			save_weights(meta["save"]);
	}

protected:
	virtual void init_weights(const std::string& info) {
		net.emplace_back(66536); // create an empty weight table with size 65536
		net.emplace_back(66536); // create an empty weight table with size 65536
		// net.emplace_back(66536);
		// net.emplace_back(66536);
		// now net.size() == 2; net[0].size() == 65536; net[1].size() == 65536
	}
	virtual void load_weights(const std::string& path) {
		std::ifstream in(path, std::ios::in | std::ios::binary);
		if (!in.is_open()) std::exit(-1);
		uint32_t size;
		in.read(reinterpret_cast<char*>(&size), sizeof(size));
		net.resize(size);
		for (weight& w : net) in >> w;
		in.close();
	}
	virtual void save_weights(const std::string& path) {
		std::ofstream out(path, std::ios::out | std::ios::binary | std::ios::trunc);
		if (!out.is_open()) std::exit(-1);
		uint32_t size = net.size();
		out.write(reinterpret_cast<char*>(&size), sizeof(size));
		for (weight& w : net) out << w;
		out.close();
	}

protected:
	std::vector<weight> net;
};

/**
 * base agent for agents with a learning rate
 */
class learning_agent : public agent {
public:
	learning_agent(const std::string& args = "") : agent(args), alpha(0.1f) {
		if (meta.find("alpha") != meta.end())
			alpha = float(meta["alpha"]);
	}
	virtual ~learning_agent() {}

protected:
	float alpha;
};

/**
 * random environment
 * add a new random tile to an empty cell
 * 2-tile: 90%
 * 4-tile: 10%
 */
class rndenv : public random_agent {
public:
	virtual void close_episode(const std::string& flag = "") {ct=0;choose =3;act=0;};
	rndenv(const std::string& args = "") : random_agent("name=random role=environment " + args),
		space({ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 }),ct(0),bag({1,2,3}), popup(0, 9) {}
	virtual action take_action(const board& after) {
		if(ct<9){
			if(choose >2){
				choose = 0;
				std::shuffle(bag.begin(), bag.end(), engine);
			}
			board::cell tile = bag[choose++];
			std::shuffle(space.begin(), space.end(), engine);
			for (int pos : space) {
				if (after(pos) != 0) {
					continue;
				}
				else{
					ct++;
					return action::place(pos, tile);
				}
			}
			return action();
		}
		
		else{
			if(choose >2){
				choose = 0;
				std::shuffle(bag.begin(), bag.end(), engine);
			}
			board::cell tile = bag[choose++];
			std::shuffle(sel_pos[act].begin(), sel_pos[act].end() ,engine);
			for (int pos =0; pos < 4; pos++) {
				if (after(sel_pos[act][pos]) != 0) continue;
				return action::place(sel_pos[act][pos], tile);
			}
			return action();
		}	
		
			
	}

private:
	std::array<int, 16> space;
	std::uniform_int_distribution<int> popup;
	int last_action;
	std::array<int, 3> bag;
	int choose = 3;
	int ct ;
	std::array<std::array<int, 4> ,4> sel_pos = {{
		{{12,13,14,15}},

		{{0,4,8,12}},
		
		{{0,1,2,3}},

		{{3,7,11,15}}
	}};

};

/**
 * dummy player
 * select a legal action randomly
 */
class player : public weight_agent {
public:
	player(const std::string& args = "") : weight_agent("name=dummy role=player " + args),
		opcode({ 0, 1, 2, 3 }) {
			for(int i=0; i<tuple_num; i++){
				net.emplace_back(weight(tile_per_tuple));
			}
		}
	virtual void open_episode(const std::string& flag = "" ) {
		count = 0;
	}
	virtual action take_action(const board& before) {

		t_tuple_feature = tuple_feature;
		board t = before;
		int next_op = select_op(t);
		int reward = t.slide(next_op);
		//train the weight if there are two board
		if(next_op != -1){
			if(count==0){
				previous = t;
				count++;
			}
			else if(count==1){
				next = t;
				train_weight(previous,next,reward,0);
				count++;
			}
			else{
				previous = next;
				next = t;
				if(reward==-1){
					train_weight(next,next,0,1);
				}
				else{
					train_weight(previous,next,reward,0);
				}
			}
			return action::slide(next_op);
		}
		return action();
	}
public:
	short select_op(const board& before){
	float max_value = -2147483648;
	board temp;
	short best_op = -1;
		for (int op = 0; op < 4; op ++) {
			temp = before;
			int reward = temp.slide(op);
			if(reward!=-1){
				if(reward + board_value(temp) > max_value){
					best_op = op;
					max_value = reward + board_value(temp);
				}
			}
		}
		act = best_op;
		return best_op;
	}
	double board_value(const board& b){
		double value = 0;
		for(int i=0; i<tuple_num; i++){
			for(int l=0; l<4; l++){
				rotate();	
				for(int m=0; m<2; m++){
					value += net[i][caculate_tuple_value(b,i)];
					reflection();	
				}
			}
		}
		return value;
	}
	/////
	unsigned long long int caculate_tuple_value(const board& b, int index_of_tuple){
		unsigned long long int tuple_value = 0;
		for(int j=0; j<num_per_tuple; j++){
			if(t_tuple_feature[index_of_tuple][j]==0){
				continue;
			}
			tuple_value |= (t_tuple_feature[index_of_tuple][j]<<4)*t_tuple_feature[index_of_tuple][j];
		}
		return tuple_value;
	}
	void train_weight(const board& previous, const board& next, int reward, int last){
		double delta = board_value(next) - board_value(previous) + reward ;
		td += delta ;
		abs_td += abs(delta);	
		double rate = (abs_td==0) ? 0.1 : td*1.0/abs_td *0.1 ;
		rate = (rate>0) ? rate : rate * (-1);
		double v_s = last ? 0 : rate * delta;
		for(int i=0; i<tuple_num; i++){
			for(int l=0; l<4; l++){
				rotate();
				for(int m=0; m<4; m++){
					net[i][caculate_tuple_value(previous,i)]+= v_s;	
					reflection();
				}
			}
		}
	}
	void reflection(){
		for(int j=0; j<4; j++){
			for (int i = 0; i < 4 ; i++) {
		  		std::swap(t_tuple_feature[j][ (i<<2)    ], t_tuple_feature[j][ (i<<2) + 3]);
		  		std::swap(t_tuple_feature[j][ (i<<2) + 1], t_tuple_feature[j][ (i<<2) + 2]);
		  	}
		}
	}
	void transpose(){
		for(int i=0; i<4; i++){
			for (int r = 0; r < 4; r++) {
				for (int c = r + 1; c < 4; c++) {
					std::swap(t_tuple_feature[i][(r<<2)+c], t_tuple_feature[i][(c<<2)+r]);
				}
			}
		}
	}
	void rotate(){
		 transpose(); reflection();
	}
private:
	std::array<int, 4> opcode;
	std::array<std::array<int, num_per_tuple> ,tuple_num> t_tuple_feature ;
	short int count = 0;
	board previous, next;	
	long long int abs_td = 0;
	long long int td = 0;
};

// void reflect_horizontal(std::array<int, 16> &arr) {
  	
//   	for (int i = 0; i < 4 ; i++) {
//   		std::swap(arr[ (i<<2)    ], arr[ (i<<2) + 3]);
//   		std::swap(arr[ (i<<2) + 1], arr[ (i<<2) + 2]);
//   	}
//   }
//   void transpose(std::array<int, 16> &arr) {
//   	for (int i = 0; i < 4; i++) 
//   		for (int j = i + 1; j < 4 ; j++) {
//   			std::swap(arr[ (i<<2) + j], arr[ (j<<2) + i]);
//   		}
//   }
//   void rotate(std::array<int, 16> &arr) {
//   	transpose(arr); reflect_horizontal(arr);
//   }	


// long get_tuple_value(const board &b, std::array<int, boardsize> arr) {
//   	long result = 0;
//   	int c = 15;
//   	std::array<int, 6> pos = {-1, -1, -1, -1, -1, -1};
//   	for(unsigned int i = 0; i < arr.size(); i++) {
//   		if ( arr[i] == 0 ) continue;
//   		pos[arr[i]-1] = i;
//   	}
//   	for ( size_t i = 0; i < arr.size(); i++ ) {
//   		result *= c;
//   		int tile = *(&b[0][0] + pos[i]);
//   		result += tile;
//   	}
//   	return result;
//   }

// static const int TUPLE_NUM = 4;
// static const int boardsize = 16;

// std::array<std::array<int, boardsize> ,TUPLE_NUM> n_tuple = {{
//   	{{1, 4, 0, 0,
//   	 2, 5, 0, 0,
//   	 3, 6, 0, 0,
//   	 0, 0, 0, 0}},

//   	{{0, 1, 4, 0,
//   	 0, 2, 5, 0,
//   	 0, 3, 6, 0,
//   	 0, 0, 0, 0}},
  	
//   	{{0, 0, 1, 0,
//   	 0, 0, 2, 0,
//   	 0, 0, 3, 0,
//   	 0, 0, 4, 0}},

//   	{{0, 0, 0, 1,
//   	 0, 0, 0, 2,
//   	 0, 0, 0, 3,
//   	 0, 0, 0, 4}}
//   }};
//   