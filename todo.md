we shoud use the same running engine for main agent or sub agents, when it spins a new agent it just calls a tag not a toolcall. remove that tool and replace with a tool that just sping the new agent with all the new details and continues streaming to the user naturally. the user sees agent spinned (with infos), then continues to see what the agent did, then sees the final report the agent sends to the orchestrator as a processing,finally the orchestrator reports to the user.
the next time, the orchestrator sees that he spawned the sub agent tag, sees all the configuration( prompt, file selection etc...) , sees the report from the agent and sees its own report to user. but not the details of what the agent did to preserve context.


## new

we need to document how an application can create a lollms client using universal profiles via the fast loading method from the configuration env, then load a personality with handbag, then inject the app's own skills and tools with the right visibility.
the agent can see that it has tools depending on their visibility (either they see exactly the full tool info, or only the name and they need to calla tool loading tag, or they see that they have a tool searching tag, and they can use to find tools, then load them. tools stay available for the whole session until the agent decides to unload them to free up some context.
the same workflow can be done with skills.
I would like a one line creation of a persona with handbag + custom tools/skills.
that would be a great addition. just one line to build the environment (lollm client), one line to build the persona powered with extra application tools and skills, and finally one line to discuss or do a task. Just pure cool ai at its finest