import random
from typing import Any, List, Tuple
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras as keras

from nptyping import NDArray


def handSize(n_players: int) -> int:
    if n_players in [2, 3]:
        return 5
    elif n_players in [4, 5]:
        return 4
    else:
        raise NameError("InvalidNPlayers")


# Global
N_PLAYER = 2
N_COLOR = 2
N_NUMBER = 3
HAND_SIZE = 3

MAX_HINTS = 5

DTILE = 2
DHANDTILE = DTILE + N_NUMBER + N_COLOR
DPLAYER = DHANDTILE * HAND_SIZE
DDECK = DTILE * (N_NUMBER * 2 * N_COLOR)

DTYPE = np.int8

# Game
INDEX = [0]
LIFES = [INDEX[-1] + 1]
HINTS = [LIFES[-1] + 1]
DECK = range(HINTS[-1] + 1, HINTS[-1] + 1 + DTILE * (N_NUMBER * 2 * N_COLOR))
DISCARDED = range(DECK[-1] + 1, DECK[-1] + 1 + (N_NUMBER * N_COLOR))
PLAYERS = range(DISCARDED[-1] + 1, DISCARDED[-1] + 1 + N_PLAYER * (DPLAYER))
PILES = range(PLAYERS[-1] + 1, PLAYERS[-1] + 1 + N_COLOR)
LAST_ROUND = [PILES[-1] + 1]

DGAME = LAST_ROUND[-1] + 1

# Action
HINT_COLOR = [INDEX[-1] + 1]
HINT_NUMBER = [HINT_COLOR[-1] + 1]
PLAYER = [HINT_NUMBER[-1] + 1]
PLAY = [PLAYER[-1] + 1]
DISCARD = [PLAY[-1] + 1]

DACTION = DISCARD[-1] + 1

HINT_ACTION = (N_PLAYER - 1) * (N_NUMBER + N_COLOR)
MAX_ACTION = HINT_ACTION + 2 * (HAND_SIZE)


# Return action
def get_hint_color_action(player: int, color: int) -> NDArray[(DACTION,), DTYPE]:
    if player not in range(1, N_PLAYER) or color not in range(0, N_COLOR):
        raise ValueError
    action = np.full((DACTION,), -1, dtype=DTYPE)
    action[HINT_COLOR] = color
    action[PLAYER] = player
    return action


def get_hint_number_action(player: int, number: int) -> NDArray[(DACTION,), DTYPE]:
    if player not in range(1, N_PLAYER) or number not in range(0, N_NUMBER):
        raise ValueError
    action = np.full((DACTION,), -1, dtype=DTYPE)
    action[HINT_NUMBER] = number
    action[PLAYER] = player
    return action


def get_play_action(tile: int) -> NDArray[(DACTION,), DTYPE]:
    if tile not in range(0, HAND_SIZE):
        raise ValueError
    action = np.full((DACTION,), -1, dtype=DTYPE)
    action[PLAY] = tile
    return action


def get_discard_action(tile: int) -> NDArray[(DACTION,), DTYPE]:
    if tile not in range(0, HAND_SIZE):
        raise ValueError
    action = np.full((DACTION,), -1, dtype=DTYPE)
    action[DISCARD] = tile
    return action


ACTION_LIST = np.empty((0, DACTION), dtype=DTYPE)
for p in range(1, N_PLAYER):
    for n in range(N_NUMBER):
        ACTION_LIST = np.vstack(
            (
                ACTION_LIST,
                get_hint_number_action(p, n),
            )
        )
    for c in range(N_COLOR):
        ACTION_LIST = np.vstack(
            (
                ACTION_LIST,
                get_hint_color_action(p, c),
            )
        )
for t in range(HAND_SIZE):
    ACTION_LIST = np.vstack((ACTION_LIST, get_discard_action(t)))
    ACTION_LIST = np.vstack((ACTION_LIST, get_play_action(t)))


# initialize
def new_deck(shuffle: bool = True) -> NDArray[(DDECK), DTYPE]:
    tiles = []
    for c in range(0, N_COLOR):
        for n in range(0, 1):
            for _ in range(3):
                tiles.append((n, c))
        for n in range(1, N_NUMBER - 1):
            for _ in range(2):
                tiles.append((n, c))
        for n in range(N_NUMBER - 1, N_NUMBER):
            for _ in range(1):
                tiles.append((n, c))
    if shuffle:
        random.shuffle(tiles)
    if 2 * len(tiles) != DDECK:
        raise NameError("Wrong lenght before conversion to np")
    deck = np.asarray([f for tile in tiles for f in tile], dtype=DTYPE)
    deck = np.reshape(deck, (DDECK,))
    return deck


def draw_initial_hand(game: NDArray[(DGAME,), DTYPE]) -> NDArray[(DGAME,), DTYPE]:
    game = game.copy()
    game[
        [
            PLAYERS[p * DPLAYER + h * DHANDTILE + i]
            for p in range(N_PLAYER)
            for h in range(HAND_SIZE)
            for i in range(DTILE)
        ]
    ] = game[DECK[: DTILE * N_PLAYER * HAND_SIZE]]
    game[DECK] = np.roll(game[DECK], -DTILE * N_PLAYER * HAND_SIZE)
    game[DECK[-DTILE * N_PLAYER * HAND_SIZE :]] = -1
    return game


def new_game(shuffle: bool = True) -> NDArray[(DGAME,), DTYPE]:
    game = np.full((DGAME,), -1, dtype=DTYPE)
    game[LIFES] = 3
    game[HINTS] = MAX_HINTS
    game[DECK] = new_deck(shuffle=shuffle)
    game[DISCARDED] = 0
    game = draw_initial_hand(game)
    return game


# action components
def draw_tile(game: NDArray[(DGAME,), DTYPE]) -> NDArray[(DGAME,), DTYPE]:
    game = game.copy()
    if (game[PLAYERS[:DPLAYER][-DHANDTILE:][:DTILE]] != -1).any():
        raise NameError("No space in hand to draw")
    if (game[LAST_ROUND] < 0 and game[DECK[:DTILE]] == -1).all():
        raise NameError("No tiles left to draw and game not ending")
    game[PLAYERS[:DPLAYER][-DHANDTILE:][:DTILE]] = game[DECK[:DTILE]]
    game[DECK] = np.roll(game[DECK], -DTILE)
    game[DECK[-DTILE:]] = -1
    return game


def remove_tile_from_hand(
    game: NDArray[(DGAME,), DTYPE], tile: int
) -> NDArray[(DGAME,), DTYPE]:
    game = game.copy()
    if tile not in range(0, HAND_SIZE):
        raise ValueError
    game[PLAYERS[:DPLAYER][tile * DHANDTILE :]] = np.roll(
        game[PLAYERS[:DPLAYER][tile * DHANDTILE :]], -DHANDTILE
    )
    game[PLAYERS[:DPLAYER][-DHANDTILE:]] = -1
    return game


def discard_tile(game: NDArray[(DGAME,), DTYPE], tile: int) -> NDArray[(DGAME,), DTYPE]:
    game = game.copy()
    if tile not in range(0, HAND_SIZE):
        raise ValueError
    if (
        game[PLAYERS[:DPLAYER][tile * DHANDTILE : (tile + 1) * DHANDTILE][:DTILE]] == -1
    ).any():
        raise NameError("Chosen position has no tile")
    game[
        DISCARDED[
            np.dot(
                game[
                    PLAYERS[:DPLAYER][tile * DHANDTILE : (tile + 1) * DHANDTILE][:DTILE]
                ],
                [N_NUMBER - 1, 1],
            )
        ]
    ] += 1
    game = remove_tile_from_hand(game, tile)
    return game


# actions
def discard(game: NDArray[(DGAME,), DTYPE], tile: int) -> NDArray[(DGAME,), DTYPE]:
    game = game.copy()
    if tile not in range(0, HAND_SIZE):
        raise ValueError
    if game[HINTS] > MAX_HINTS:
        raise NameError("Max hints avaible, discard forbidden")
    if (
        game[PLAYERS[:DPLAYER][tile * DHANDTILE : (tile + 1) * DHANDTILE][:DTILE]] == -1
    ).any():
        raise NameError("Chosen position has no tile")
    game[HINTS] += 1
    game = discard_tile(game, tile)
    game = draw_tile(game)
    return game


def play(game: NDArray[(DGAME,), DTYPE], tile: int) -> NDArray[(DGAME,), DTYPE]:
    game = game.copy()
    if tile not in range(0, HAND_SIZE):
        raise ValueError
    if game[LIFES] <= 0:
        raise NameError("No lifes left, game should be already ended")
    if (
        game[PLAYERS[:DPLAYER][tile * DHANDTILE : (tile + 1) * DHANDTILE][:DTILE]] == -1
    ).any():
        raise NameError("Chosen position has no tile")
    number, color = game[
        PLAYERS[:DPLAYER][tile * DHANDTILE : (tile + 1) * DHANDTILE][:DTILE]
    ]
    if game[PILES[color]] == number - 1:
        game[PILES[color]] += 1
        game = remove_tile_from_hand(game, tile)
    else:
        game[LIFES] -= 1
        game = discard_tile(game, tile)
    game = draw_tile(game)
    return game


def hint_number(game: NDArray[(DGAME,), DTYPE], player: int, number: int):
    game = game.copy()
    if player not in range(1, N_PLAYER):
        raise ValueError
    if number not in range(0, N_NUMBER):
        raise ValueError
    if game[HINTS] <= 0:
        raise NameError("No hints left, hint forbidden")
    game[
        [
            PLAYERS[player * DPLAYER : (player + 1) * DPLAYER][
                t * DHANDTILE : (t + 1) * DHANDTILE
            ][DTILE + number]
            for t in range(HAND_SIZE)
        ]
    ] = np.where(
        game[
            [
                PLAYERS[player * DPLAYER : (player + 1) * DPLAYER][
                    t * DHANDTILE : (t + 1) * DHANDTILE
                ][0]
                for t in range(HAND_SIZE)
            ]
        ]
        == number,
        1,
        0,
    )
    game[HINTS] -= 1
    return game


def hint_color(game: NDArray[(DGAME,), DTYPE], player: int, color: int):
    game = game.copy()
    if player not in range(1, N_PLAYER):
        raise ValueError
    if color not in range(0, N_COLOR):
        raise ValueError
    if game[HINTS] <= 0:
        raise NameError("No hints left, hint forbidden")
    game[
        [
            PLAYERS[player * DPLAYER : (player + 1) * DPLAYER][
                t * DHANDTILE : (t + 1) * DHANDTILE
            ][DTILE + N_NUMBER + color]
            for t in range(HAND_SIZE)
        ]
    ] = np.where(
        game[
            [
                PLAYERS[player * DPLAYER : (player + 1) * DPLAYER][
                    t * DHANDTILE : (t + 1) * DHANDTILE
                ][1]
                for t in range(HAND_SIZE)
            ]
        ]
        == color,
        1,
        0,
    )
    game[HINTS] -= 1
    return game


# utils
def score(game: NDArray[(DGAME,), DTYPE]) -> int:
    return np.sum(game[PILES], dtype=DTYPE) + N_COLOR


# def vscore(games: NDArray[(Any, DGAME), DTYPE]) -> NDArray[(Any,), DTYPE]:
#     return np.apply_along_axis(score, 1, games)


def is_ended(game: NDArray[(DGAME,), DTYPE]) -> bool:
    if game[LIFES] == 0:
        return True
    if game[LAST_ROUND] == 0:
        return True
    if score(game) == N_NUMBER * N_COLOR:
        return True
    return False


# def vis_ended(games: NDArray[(Any, DGAME), DTYPE]) -> NDArray[(Any,), bool]:
#     return np.apply_along_axis(is_ended, 1, games)


def next_turn(game: NDArray[(DGAME,), DTYPE]) -> NDArray[(DGAME,), DTYPE]:
    game = game.copy()
    game[PLAYERS] = np.roll(game[PLAYERS], DPLAYER)
    if (game[DECK] == -1).all():
        if game[LAST_ROUND] == -1:
            game[LAST_ROUND] = N_PLAYER
        elif game[LAST_ROUND] > 0:
            game[LAST_ROUND] -= 1
        elif game[LAST_ROUND] == 0:
            raise NameError("Last round already completed, game should be alredy ended")
    return game


# move
def available_moves(game: NDArray[(DGAME,), DTYPE]) -> List[int]:
    actions: List[int] = []
    if game[HINTS] > 0:
        actions = actions + list(range(HINT_ACTION))
    for t in range(HAND_SIZE):
        if (
            game[PLAYERS[:DPLAYER][t * DHANDTILE : (t + 1) * DHANDTILE][:DTILE]] != -1
        ).all():
            if game[HINTS] < MAX_HINTS:
                actions = actions + [HINT_ACTION + 2 * t]
            actions = actions + [HINT_ACTION + 2 * t + 1]
    return actions


# def vavailable_moves(games: NDArray[(Any, DGAME), DTYPE]) -> NDArray[(Any), object]:
#     return np.apply_along_axis(available_moves, 1, games)


def choose_action(
    game: NDArray[(DGAME,), DTYPE], epsilon: float, model: keras.Model, INPUT: List[int]
) -> int:
    actions = available_moves(game)
    if np.random.random() > epsilon:  # exploit
        return actions[
            np.asscalar(
                np.argmax(
                    model.predict(np.expand_dims(game, axis=0)[:, INPUT])[:, actions],
                    axis=1,
                )
            )
        ]
    else:  # explore
        return random.choice(actions)


# def vchoose_action(
#     games: NDArray[(Any, DGAME), DTYPE],
#     epsilon: float,
#     model: keras.Model,
#     INPUT: List[int],
# ) -> NDArray[(Any, 1), int]:
#     return np.apply_along_axis(choose_action, 1, games, epsilon, model, INPUT)


def move(game: NDArray[(DGAME,), DTYPE], action: int) -> NDArray[(DGAME,), DTYPE]:
    if ACTION_LIST[action][HINT_NUMBER] != -1:
        return next_turn(
            hint_number(
                game,
                np.asscalar(ACTION_LIST[action][PLAYER]),
                np.asscalar(ACTION_LIST[action][HINT_NUMBER]),
            )
        )
    if ACTION_LIST[action][HINT_COLOR] != -1:
        return next_turn(
            hint_color(
                game,
                np.asscalar(ACTION_LIST[action][PLAYER]),
                np.asscalar(ACTION_LIST[action][HINT_COLOR]),
            )
        )
    if ACTION_LIST[action][PLAY] != -1:
        return next_turn(play(game, np.asscalar(ACTION_LIST[action][PLAY])))
    if ACTION_LIST[action][DISCARD] != -1:
        return discard(game, np.asscalar(ACTION_LIST[action][DISCARD]))


# def vmove(
#     games: NDArray[(Any, DGAME), DTYPE], actions: NDArray[(Any, 1), int]
# ) -> NDArray[(Any, DGAME), DTYPE]:
#     return np.apply_along_axis(move, 1, games, actions)


# for learning
def get_memories(
    memories: NDArray[(Any, DGAME + DACTION + DGAME), DTYPE],
    n: int,
) -> NDArray[(Any, DGAME), DTYPE]:
    if n == 0:
        return np.empty((0, DGAME), dtype=DTYPE)
    if memories.shape[0] > 0:
        memories = memories[np.argsort(np.apply_along_axis(score, 1, memories), axis=0)]
    if memories.shape[0] < n:
        return memories
    else:
        return memories[-n:, :]


def get_games(
    memories: NDArray[(Any, DGAME), DTYPE],
    new: int,
    old: int,
    sample_memories: bool = True,
) -> NDArray[(Any, DGAME), DTYPE]:
    games = np.empty((0, DGAME), dtype=DTYPE)
    for _ in range(new):
        games = np.vstack((games, new_game()))
    if memories.shape[0] < old:
        games = np.vstack((games, memories))
    else:
        if sample_memories:
            games = np.vstack(
                (
                    games,
                    memories[
                        np.random.choice(memories.shape[0], size=old, replace=False), :
                    ],
                )
            )
        else:
            games = np.vstack(
                (
                    games,
                    memories[-old:, :],
                )
            )
    return games


def try_all_moves(
    game: NDArray[(DGAME,), DTYPE]
) -> NDArray[(MAX_ACTION, DGAME), DTYPE]:
    post = np.full((MAX_ACTION, DGAME), -1, dtype=DTYPE)
    if is_ended(game):
        post[:, 0] = 0
        return post
    else:
        actions = available_moves(game)
        forbidden = np.ones(MAX_ACTION, dtype=bool)
        forbidden[actions] = False
        post[forbidden, 0] = 0
        post[actions, :] = np.apply_along_axis(
            lambda a: move(game, np.asscalar(a)), 1, np.array(actions).reshape(-1, 1)
        )
        return post


# def vtry_all_moves(games: NDArray[(Any, DGAME), DTYPE]) -> NDArray[(Any, MAX_ACTION, DGAME), DTYPE]:
#     return np.apply_along_axis(try_all_moves, 1, games)


def Q(
    memories: NDArray[(Any, DGAME), DTYPE],
    gamma: float,
    model: keras.Model,
    INPUT: List[int],
) -> NDArray[(Any, 1), float]:
    post = np.apply_along_axis(try_all_moves, 1, memories)
    if gamma < 1e-6:
        maxQ = 1
        Qs = np.multiply(
            (~np.apply_along_axis(is_ended, 1, memories)).astype(int).reshape(-1, 1),
            np.apply_along_axis(score, 2, post)
            - np.apply_along_axis(score, 1, memories).reshape(-1, 1),
        )
    else:
        maxQ = (1 - gamma ** (N_NUMBER * N_COLOR)) / (1 - gamma)
        Qs = np.multiply(
            (~np.apply_along_axis(is_ended, 1, memories)).astype(int).reshape(-1, 1),
            np.apply_along_axis(score, 2, post)
            - np.apply_along_axis(score, 1, memories).reshape(-1, 1)
            + gamma
            * np.reshape(
                np.multiply(
                    np.amax(
                        model.predict(
                            np.reshape(post, (-1, DGAME))[:, INPUT], batch_size=512
                        ),
                        axis=1,
                    ).reshape(-1, 1),
                    -post[:, :, 0].reshape(-1, 1),
                ),
                (memories.shape[0], MAX_ACTION),
            ),
        )
    Qs[Qs < 0] = 0
    Qs[Qs > maxQ] = maxQ
    return Qs


# for workflow
def play_game(
    epsilon: float,
    memory_size: int,
    new: int,
    old: int,
    memories: NDArray[(Any, DGAME + DACTION + DGAME), DTYPE],
    points: NDArray[(Any, 4), float],
    model: keras.Model,
    INPUT: List[int],
    sample_memories: bool = True,
) -> Tuple[NDArray[(Any, DGAME + DACTION + DGAME), DTYPE], NDArray[(Any, 4), float]]:
    memories = get_memories(memories, memory_size)
    games = get_games(memories, new, old, sample_memories=sample_memories)
    while (~(np.apply_along_axis(is_ended, 1, games))).all():
        memories = np.vstack(
            (memories, games[~(np.apply_along_axis(is_ended, 1, games))])
        )
        games[~(np.apply_along_axis(is_ended, 1, games))] = np.apply_along_axis(
            lambda x: move(x[1:], x[0]),
            1,
            np.insert(
                games,
                0,
                np.apply_along_axis(
                    choose_action, 1, games, epsilon, model, INPUT
                ).transpose(),
                axis=1,
            ),
        )
    memories = np.vstack((memories, games))
    scores = np.apply_along_axis(score, 1, games)
    points = np.vstack(
        (
            points,
            np.array(
                [np.amin(scores), np.mean(scores), np.median(scores), np.max(scores)]
            ),
        ),
    )
    print(scores)
    return memories, points


def train(
    gamma: float,
    memories: NDArray[(Any, DGAME + DACTION + DGAME), DTYPE],
    loss: NDArray[(Any, 1), float],
    model: keras.Model,
    INPUT: List[int],
    patience: int = 500,
    name: str = "hanabi",
) -> NDArray[(Any, 1), float]:
    # np.random.shuffle(memories)
    loss = np.vstack(
        (
            loss,
            np.nanmin(
                model.fit(
                    x=memories[:, INPUT],
                    y=Q(memories, gamma, model, INPUT),
                    epochs=5000,
                    # validation_split=0.2,
                    batch_size=32,
                    callbacks=[
                        keras.callbacks.EarlyStopping(
                            monitor="loss",
                            min_delta=0.01,
                            patience=patience,
                            restore_best_weights=False,
                        ),
                        keras.callbacks.TensorBoard(log_dir=name + "/logs"),
                    ],
                ).history["loss"]
            ),
        )
    )
    return loss


def save(
    memories: NDArray[(Any, DGAME + DACTION + DGAME), DTYPE],
    points: NDArray[(Any, 4), float],
    loss: NDArray[(Any, 1), float],
    model: keras.Model,
    Qs: NDArray[(Any, 4), float],
    path: str,
    last: bool = False,
    first: bool = False,
):
    if not os.path.exists(path):
        os.mkdir(path)
    if last:
        np.save(os.path.join(path, "last_memories.npy"), memories)
        np.save(os.path.join(path, "last_points.npy"), points)
    elif first:
        np.save(os.path.join(path, "memories_t0.npy"), memories)
        np.save(os.path.join(path, "loss_t0.npy"), loss)
        keras.models.save_model(model, os.path.join(path, "nn_t0"))
    else:
        np.save(os.path.join(path, "memories.npy"), memories)
        np.save(os.path.join(path, "points.npy"), points)
        np.save(os.path.join(path, "loss.npy"), loss)
        np.save(os.path.join(path, "Qs.npy"), Qs)
        keras.models.save_model(model, os.path.join(path, "nn"))


def plot(
    loss: NDArray[(Any, 1), float],
    points: NDArray[(Any, 4), float],
    Qs: NDArray[(Any, 4), float],
    path: str,
):
    plt.plot(loss)
    plt.title("Loss")
    plt.savefig(os.path.join(path, "loss.png"))
    plt.show()

    plt.plot(points[:, 0])
    plt.plot(points[:, 1])
    plt.plot(points[:, 2])
    plt.plot(points[:, 3])
    plt.title("Points")
    plt.legend(["min", "mean", "median", "max"])
    plt.savefig(os.path.join(path, "points.png"))
    plt.show()

    plt.plot(Qs[:, 0])
    plt.plot(Qs[:, 1])
    plt.plot(Qs[:, 2])
    plt.plot(Qs[:, 3])
    plt.title("Q")
    plt.legend(["min", "mean", "median", "max"])
    plt.savefig(os.path.join(path, "Qs.png"))
    plt.show()


def print_game(game) -> None:
    print("LIFES ", game[LIFES])
    print("HINTS ", game[HINTS])
    print("DECK")
    print(game[DECK])
    print("DISCARDED")
    print(game[DISCARDED])
    for p in range(N_PLAYER):
        print("PLAYER ", p + 1)
        for t in range(HAND_SIZE):
            print(
                game[PLAYERS][p * DPLAYER : (p + 1) * DPLAYER][
                    t * DHANDTILE : (t + 1) * DHANDTILE
                ]
            )
    print("PILES ", game[PILES])


def print_action(action) -> None:
    print("HINT")
    print("Number ", action[HINT_NUMBER])
    print("Color ", action[HINT_COLOR])
    print("Player ", action[PLAYER])
    print("PLAY")
    print("Tile ", action[PLAY])
    print("DISCARD")
    print("Tile ", action[DISCARD])
