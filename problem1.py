from agents4e import ReflexVacuumAgent, Agent, SimpleReflexAgentProgram, TrivialVacuumEnvironment, loc_A, loc_B


def RuleVacuumAgent1b(rules):
    program = SimpleReflexAgentProgram(rules)

    return Agent(program)


if __name__ == "__main__":

    print("------------------")
    print("1a")
    print("------------------")
    for a_status in ["Clean", "Dirty"]:
        for b_status in ["Clean", "Dirty"]:
            for start in [loc_A, loc_B]:
                e = TrivialVacuumEnvironment()
                a = ReflexVacuumAgent()

                e.status[loc_A] = a_status
                e.status[loc_B] = b_status
                a.location = start
                e.add_thing(a)

                print(f"Loc A is {a_status} | Loc B is {b_status} | Start at Loc {start}")
                e.run(5)
                
    print("")                
    print("------------------")
    print("1b")
    print("------------------")

    rules = {
                (loc_A, 'Clean'): 'Right',
                (loc_B, 'Clean'): 'Left',
                (loc_A, 'Dirty'): 'Suck',
                (loc_B, 'Dirty'): 'Suck'
            }

    for a_status in ["Clean", "Dirty"]:
        for b_status in ["Clean", "Dirty"]:
            for start in [loc_A, loc_B]:
                e = TrivialVacuumEnvironment()
                a = RuleVacuumAgent1b(rules)

                e.status[loc_A] = a_status
                e.status[loc_B] = b_status
                a.location = start
                e.add_thing(a)

                print(f"Loc A is {a_status} | Loc B is {b_status} | Start at Loc {start}")
                e.run(5)
