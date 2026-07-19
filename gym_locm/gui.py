import pygame
from gym_locm.engine import Phase, Creature, GreenItem, RedItem, BlueItem

class GUIRenderer:
    def __init__(self):
        self.screen = None
        self.font = None
        self.font_small = None
        self.font_title = None
        self.font_large = None
        self.clock = None
        # Track positions for animations
        self.card_positions = {}
        self.last_frame_cards = {}
        self.fading_cards = {}

    def init_pygame(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((1280, 800))
            pygame.display.set_caption("LOCM Environment")
            self.font = pygame.font.SysFont("Arial", 16)
            self.font_small = pygame.font.SysFont("Arial", 12)
            self.font_title = pygame.font.SysFont("Arial", 24, bold=True)
            self.font_large = pygame.font.SysFont("Arial", 36, bold=True)
            self.clock = pygame.time.Clock()

    def render(self, state):
        self.init_pygame()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        # Background
        self.screen.fill((240, 240, 240))

        if state.phase == Phase.DECK_BUILDING:
            self._draw_deck_building(state)
        elif state.phase == Phase.BATTLE:
            self._animate_and_draw_battle(state)
        elif state.phase == Phase.ENDED:
            self._draw_ended(state)

        pygame.display.flip()
        pygame.time.delay(1500)

    def _draw_card(self, x, y, card, is_board=False, alpha=255, is_face_down=False):
        # Card dimensions
        card_w = 100
        card_h = 140

        surf = pygame.Surface((card_w, card_h), pygame.SRCALPHA)

        if is_face_down:
            pygame.draw.rect(surf, (100, 100, 150), (0, 0, card_w, card_h))
            pygame.draw.rect(surf, (0, 0, 0), (0, 0, card_w, card_h), 3)
            if alpha < 255:
                surf.set_alpha(alpha)
            self.screen.blit(surf, (x, y))
            return

        # Colors based on type
        if isinstance(card, Creature):
            border_color = (200, 200, 0)
        elif isinstance(card, GreenItem):
            border_color = (0, 200, 0)
        elif isinstance(card, RedItem):
            border_color = (200, 0, 0)
        elif isinstance(card, BlueItem):
            border_color = (0, 0, 200)
        else:
            border_color = (100, 100, 100)

        # Background
        pygame.draw.rect(surf, (255, 255, 255), (0, 0, card_w, card_h))
        # Border
        pygame.draw.rect(surf, border_color, (0, 0, card_w, card_h), 4)

        # Name
        name_text = self.font_small.render(card.name[:15], True, (0, 0, 0))
        surf.blit(name_text, (5, 5))

        # Cost
        pygame.draw.circle(surf, (0, 0, 255), (card_w - 15, 15), 12)
        cost_text = self.font.render(str(card.cost), True, (255, 255, 255))
        surf.blit(cost_text, (card_w - 20, 7))

        # Attack / Defense
        pygame.draw.circle(surf, (255, 200, 0), (15, card_h - 15), 12)
        atk_text = self.font.render(str(card.attack), True, (0, 0, 0))
        surf.blit(atk_text, (10, card_h - 23))

        pygame.draw.rect(surf, (200, 0, 0), (card_w - 25, card_h - 25, 20, 20))
        def_text = self.font.render(str(card.defense), True, (255, 255, 255))
        surf.blit(def_text, (card_w - 20, card_h - 23))

        # Text/Keywords
        kw = ""
        for a in "BCDXGLW":
            try:
                if card.has_ability(a):
                    kw += a
            except KeyError:
                pass

        if kw:
            kw_text = self.font_small.render(kw, True, (0, 0, 0))
            surf.blit(kw_text, (5, 40))

        if is_board and isinstance(card, Creature):
            if card.can_attack and not getattr(card, 'has_attacked_this_turn', False):
                # Green border inner if can attack
                pygame.draw.rect(surf, (0, 255, 0), (4, 4, card_w-8, card_h-8), 2)
                
        if alpha < 255:
            surf.set_alpha(alpha)
            
        self.screen.blit(surf, (x, y))


    def _draw_deck_building(self, state):
        self.screen.fill((240, 240, 240))
        title = self.font_title.render(f"Draft Phase - Turn {state.turn} - Player {state.current_player.id}", True, (0, 0, 0))
        self.screen.blit(title, (20, 20))

        start_x = 1280 // 2 - (len(state.current_player.hand) * 110) // 2
        y = 300
        for i, card in enumerate(state.current_player.hand):
            self._draw_card(start_x + i * 110, y, card)
            idx_text = self.font.render(f"Pick [{i}]", True, (0, 0, 0))
            self.screen.blit(idx_text, (start_x + i * 110 + 20, y - 25))

        pygame.display.flip()
        pygame.time.delay(100)

    def _animate_and_draw_battle(self, state):
        player = state.current_player
        opponent = state.opposing_player

        # Target positions for current frame
        targets = {}

        # Opponent Board Targets
        for lane_idx, cards in enumerate(opponent.lanes):
            start_x = 100 + lane_idx * (1280//2)
            y = 150
            for i, card in enumerate(cards):
                targets[card.instance_id] = (start_x + i * 110, y, card, True, False)

        # Player Board Targets
        for lane_idx, cards in enumerate(player.lanes):
            start_x = 100 + lane_idx * (1280//2)
            y = 400
            for i, card in enumerate(cards):
                targets[card.instance_id] = (start_x + i * 110, y, card, True, False)

        # Player Hand Targets
        hand_start_x = 10
        hand_y = 650
        for i, card in enumerate(player.hand):
            targets[card.instance_id] = (hand_start_x + i * 105, hand_y, card, False, False)
            
        # Opponent Hand Targets
        opp_hand_start_x = 10
        opp_hand_y = 10
        for i, card in enumerate(opponent.hand):
            targets[card.instance_id] = (opp_hand_start_x + i * 105, opp_hand_y, card, False, True)
            
        # Initialize any new cards in self.card_positions to a starting point
        for cid, (tx, ty, card, is_b, is_fd) in targets.items():
            if cid not in self.card_positions:
                # If going to opponent board, spawn from opponent deck/hand area
                if ty < 300:
                    self.card_positions[cid] = (1000, 55)
                else:
                    self.card_positions[cid] = (1000, 650)
                    
        # Identify newly dead cards
        for cid, (tx, ty, card, is_b, is_fd) in self.last_frame_cards.items():
            if cid not in targets and is_b and isinstance(card, Creature):
                if cid in self.card_positions:
                    cx, cy = self.card_positions[cid]
                    self.fading_cards[cid] = {'x': cx, 'y': cy, 'card': card, 'alpha': 255}

        # Check for attack animations by looking at the last action taken
        attacking_card_id = None
        attack_target_id = None
        
        last_action = None
        if len(player.actions) > 0:
            last_action = player.actions[-1]
        elif len(opponent.actions) > 0:
            last_action = opponent.actions[-1]
            
        if last_action and getattr(last_action, 'type', None) is not None:
            if last_action.type.name == 'ATTACK':
                attacking_card_id = last_action.origin
                attack_target_id = last_action.target

        # Animation Loop (30 frames)
        frames = 30
        for frame in range(frames):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return

            self.screen.fill((240, 240, 240))

            # Draw static elements (Hero boxes, lines)
            self._draw_battle_statics(player, opponent)

            # Draw fading cards
            for cid in list(self.fading_cards.keys()):
                info = self.fading_cards[cid]
                info['alpha'] = max(0, info['alpha'] - (255 / frames))
                if info['alpha'] > 0:
                    bump_y = 0
                    if cid == attacking_card_id:
                        if frame < 15:
                            bump_y = -50 if info['y'] > 300 else 50
                    elif cid == attack_target_id:
                        if frame < 15:
                            bump_y = 50 if info['y'] > 300 else -50
                            
                    self._draw_card(info['x'], info['y'] + bump_y, info['card'], is_board=True, alpha=int(info['alpha']))

            # Interpolate and draw cards
            for cid, (tx, ty, card, is_b, is_fd) in targets.items():
                cx, cy = self.card_positions[cid]
                # Linear interpolation
                nx = cx + (tx - cx) * 0.2
                ny = cy + (ty - cy) * 0.2
                
                # Attack bump logic: if this card is attacking, it bumps forward halfway through the animation
                # If this card is the target, it bumps backward
                bump_y = 0
                if cid == attacking_card_id and is_b:
                    if frame < 15:
                        bump_y = -50 if ty > 300 else 50 # Move towards center
                elif cid == attack_target_id and is_b:
                    if frame < 15:
                        bump_y = 50 if ty > 300 else -50 # Move away from center
                        
                # Snap to target if very close to avoid jitter
                if abs(nx - tx) < 1: nx = tx
                if abs(ny - ty) < 1: ny = ty
                
                self.card_positions[cid] = (nx, ny)
                self._draw_card(nx, ny + bump_y, card, is_board=is_b, is_face_down=is_fd)
                
            pygame.display.flip()
            self.clock.tick(60)
            
        # Hold for a moment to let the user see the result of the action
        pygame.time.delay(100)

        # Clean up old cards from tracking dict to avoid memory leak over thousands of steps
        current_ids = set(targets.keys())
        self.card_positions = {k: v for k, v in self.card_positions.items() if k in current_ids}
        self.fading_cards = {k: v for k, v in self.fading_cards.items() if v['alpha'] > 0}
        self.last_frame_cards = targets


    def _draw_battle_statics(self, player, opponent):
        # Turn Indicator (Center Top)
        turn_indicator = self.font_large.render(f"Player {player.id}'s Turn", True, (0, 0, 0))
        self.screen.blit(turn_indicator, (1280//2 - 100, 10))

        # Opponent section (Top Right)
        pygame.draw.rect(self.screen, (200, 150, 150), (800, 10, 400, 80))
        pygame.draw.rect(self.screen, (0, 0, 0), (800, 10, 400, 80), 3)
        opp_stats = f"Opponent | {opponent.health} HP | {opponent.mana}/{opponent.base_mana} MP"
        self.screen.blit(self.font_title.render(opp_stats, True, (0, 0, 0)), (820, 20))
        opp_cards = f"Hand: {len(opponent.hand)} | Deck: {len(opponent.deck)}"
        self.screen.blit(self.font.render(opp_cards, True, (0, 0, 0)), (820, 55))

        # Board Lane Boxes
        lane_width = 1280 // 2
        pygame.draw.line(self.screen, (0, 0, 0), (lane_width, 100), (lane_width, 550), 2)
        pygame.draw.rect(self.screen, (0, 0, 0), (10, 100, 1260, 450), 2)
        pygame.draw.line(self.screen, (0, 0, 0), (0, 360), (1280, 360), 2) # mid line
        pygame.draw.line(self.screen, (0, 0, 0), (0, 560), (1280, 560), 2)

        self.screen.blit(self.font.render("Left Lane", True, (100, 100, 100)), (10, 340))
        self.screen.blit(self.font.render("Right Lane", True, (100, 100, 100)), (1280//2 + 10, 340))

        # Player section (Bottom Right)
        pygame.draw.rect(self.screen, (150, 150, 200), (800, 650, 400, 80))
        pygame.draw.rect(self.screen, (0, 0, 0), (800, 650, 400, 80), 3)
        player_stats = f"Player {player.id} | {player.health} HP | {player.mana}/{player.base_mana} MP"
        self.screen.blit(self.font_title.render(player_stats, True, (0, 0, 0)), (820, 660))
        player_cards = f"Hand: {len(player.hand)} | Deck: {len(player.deck)}"
        self.screen.blit(self.font.render(player_cards, True, (0, 0, 0)), (820, 695))

    def _draw_ended(self, state):
        self.screen.fill((240, 240, 240))
        title = self.font_title.render(f"Game Over! Player {state.winner} won!", True, (0, 100, 0))
        self.screen.blit(title, (1280//2 - 150, 800//2 - 50))
        pygame.display.flip()
        pygame.time.delay(1500)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.font = None
            self.font_small = None
            self.font_title = None
            self.font_large = None
            self.clock = None
