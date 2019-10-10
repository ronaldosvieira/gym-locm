#include "engine.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

int instance_counter = 0;
Card _cards[160];

/* Card methods */
int next_instance_id() {
    return ++instance_counter;
}

void load_cards() {
    FILE* card_list;

    card_list = fopen("../gym_locm/cardlist.txt", "r");

    if (card_list == NULL) {
        printf("Error: cardlist.txt could not be opened.\n");
        exit(EXIT_FAILURE);
    }

    char* format = "%d ; %d ; %d ; %d ; %d ; %6s ; %d ; %d ; %d\n";

    for (int i = 0; i < 160; i++) {
        char card_keywords[7];

        _cards[i].instance_id = -1;

        fscanf(card_list, format, &_cards[i].id, &_cards[i].type, &_cards[i].cost,
                &_cards[i].attack, &_cards[i].defense, card_keywords,
                &_cards[i].player_hp, &_cards[i].enemy_hp, &_cards[i].card_draw);

        _cards[i].keywords = 0;

        for (int j = 0; j < 6; j++) {
            if (card_keywords[j] != '-') {
                _cards[i].keywords += 1 << j; // NOLINT(hicpp-signed-bitwise)
            }
        }

        _cards[i].can_attack = 0;
        _cards[i].has_attacked = 0;
        _cards[i].passed = 0;
    }

    fclose(card_list);
}

int has_keyword(Card card, Keyword keyword) {
    return card.keywords >> keyword & 1; // NOLINT(hicpp-signed-bitwise)
}

void add_keyword(Card* card, Keyword keyword) {
    card->keywords |= 1 << keyword; // NOLINT(hicpp-signed-bitwise)
}

void remove_keyword(Card* card, Keyword keyword) {
    card->keywords &= ~(1 << keyword); // NOLINT(hicpp-signed-bitwise)
}

Card* copy_card(Card card) {
    Card *copied_card = malloc(sizeof(Card));

    memcpy(copied_card, &card, sizeof(Card));

    return copied_card;
}

/* Player methods */
Player* new_player(PlayerId id, int health, int mana,
        int cards_remaining, int next_rune, int bonus_draw) {
    Player* player = malloc(sizeof(Player));

    player->id = id;
    player->health = health;
    player->mana = mana;
    player->cards_remaining = cards_remaining;
    player->next_rune = next_rune;
    player->bonus_draw = bonus_draw;

    return player;
}

void damage_player(Player* player, int amount) {
    player->health -= amount;

    while (player->health <= player->next_rune) {
        player->next_rune -= 5;
        player->bonus_draw += 1;
    }
}

/* State methods */
State* new_state(int turn, Player* p0, Player* p1, Card *cards) {
    State* state = malloc(sizeof(State));

    state->turn = turn;
    state->players[FIRST_PLAYER] = p0;
    state->players[SECOND_PLAYER] = p1;
    state->cards = cards;
    state->available_actions = NULL;

    return state;
}

void act(State* state, Action action) {
    if (action.type == PASS)
        _do_pass(state);
    else if (action.type == SUMMON)
        _do_summon(state, action.origin, action.target);
    else if (action.type == USE)
        _do_use(state, action.origin, action.target);
    else if (action.type == ATTACK)
        _do_attack(state, action.origin, action.target);
}

void _do_pass(State* state) {
    // todo: implement
}

void _do_summon(State* state, int origin_id, Lane lane) {
    // todo: implement
}

void _do_use(State* state, int origin_, int target_id) {
    // todo: implement
}

void _do_attack(State* state, int origin_id, int target_id) {
    // todo: implement
}

void _get_available_actions(State* state) {
    // todo: implement
}

int main() {
    struct timeval time;
    gettimeofday(&time, NULL);

    srand((time.tv_sec * 1000) + (time.tv_usec / 1000));

    load_cards();
}