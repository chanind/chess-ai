from dataclasses import dataclass
from thespian.actors import Actor, ActorExitRequest
import torch
from tqdm import tqdm

from .SelfPlayGameActor import PlayGameMessage, PlayGameResult, SelfPlayGameActor


@dataclass
class PlayGamesMessage:
    mcts_simulations: int
    temp_threshold: int
    device: torch.device
    model_predict_actor_addr: any
    num_games: int


class SelfPlayGamesManagerActor(Actor):
    """
    Helper actor to manage the running of the individual game actors in parallel, and collect their results
    """

    def receiveMessage(self, msg, sender):
        if isinstance(msg, PlayGamesMessage):
            print("PLAYING GAMES")
            self.play_games(
                mcts_simulations=msg.mcts_simulations,
                temp_threshold=msg.temp_threshold,
                device=msg.device,
                model_predict_actor_addr=msg.model_predict_actor_addr,
                num_games=msg.num_games,
            )
            self.play_games_sender = sender
        if isinstance(msg, PlayGameResult):
            # got a game result!
            print("GAME RESULT")
            self.handle_game_result(msg.train_examples, msg.pgn)

    def handle_game_result(self, train_examples, pgn):
        self.training_examples.append(train_examples)
        self.pgns.append(pgn)
        self.pbar.update(n=1)
        if len(self.training_examples) >= len(self.individual_games_actors):
            self.pbar.close()
            self.send(self.play_games_sender, self.training_examples, self.pgns)
            for actor in self.individual_games_actors:
                self.send(actor, ActorExitRequest)
            self.individual_games_actors = None
            self.pbar = None

    def play_games(
        self,
        mcts_simulations: int,
        temp_threshold: int,
        device: torch.device,
        model_predict_actor_addr: any,
        num_games: int,
    ):
        self.pbar = tqdm(total=num_games, unit="game")
        self.individual_games_actors = [
            self.createActor(SelfPlayGameActor) for _ in range(num_games)
        ]
        self.training_examples = []
        self.pgns = []
        for game_actor in self.individual_games_actors:
            self.send(
                game_actor,
                PlayGameMessage(
                    mcts_simulations=mcts_simulations,
                    temp_threshold=temp_threshold,
                    device=device,
                    model_predict_actor_addr=model_predict_actor_addr,
                ),
            )
