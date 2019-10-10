#ifndef GYM_LOCM_ENGINE_H
#define GYM_LOCM_ENGINE_H

#define PASS 0
#define PICK 1
#define SUMMON 2
#define USE 3
#define ATTACK 4

typedef int ActionType;

typedef struct Action {
    ActionType type;
    int origin, target;
} Action;

typedef struct ActionList {
    Action *action;
    struct ActionList *next;
} ActionList;

#define CREATURE 0
#define GREEN_ITEM 1
#define RED_ITEM 2
#define BLUE_ITEM 3

typedef int CardType;

#define OUTSIDE -1
#define P0_HAND 0
#define P0_BOARD 1
#define P1_BOARD 2
#define P1_HAND 3

typedef int Location;

#define LEFT_LANE 0
#define RIGHT_LANE 0

typedef int Lane;

#define BREAKTHROUGH 0
#define CHARGE 1
#define DRAIN 2
#define GUARD 3
#define LETHAL 4
#define WARD 5

typedef int Keyword;

typedef struct Card {
    int id, instance_id;
    CardType type;
    int cost, attack, defense, player_hp, enemy_hp, card_draw;
    int keywords;
    Location location;
    Lane lane;

    int can_attack, has_attacked;
    int passed;
} Card;

#define FIRST_PLAYER 0
#define SECOND_PLAYER 1

typedef int PlayerId;

typedef struct Player {
    PlayerId id;
    int health;
    int mana;
    int cards_remaining;
    int next_rune;
    int bonus_draw;
} Player;

typedef struct State {
    int turn;

    PlayerId current_player;
    Player *players[2];

    Card *cards;

    ActionList *available_actions;
} State;

/* Card methods */
int next_instance_id();
void load_cards();

int has_keyword(Card card, Keyword keyword);
void add_keyword(Card* card, Keyword keyword);
void remove_keyword(Card* card, Keyword keyword);
Card* copy_card(Card card);

/* Player methods */
Player* new_player(PlayerId id, int health, int mana,
        int cards_remaining, int next_rune, int bonus_draw);
void damage_player(Player* player, int amount);

/* State methods */
State* new_state(int turn, Player* p0, Player* p1, Card *cards);

void act(State* state, Action action);
void _do_pass(State* state);
void _do_summon(State* state, int origin_id, Lane lane);
void _do_use(State* state, int origin_, int target_id);
void _do_attack(State* state, int origin_id, int target_id);

void _get_available_actions(State* state);

#endif //GYM_LOCM_ENGINE_H