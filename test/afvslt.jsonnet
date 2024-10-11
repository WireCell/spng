{
    gpu: true,
    seed: 123456,
    arrays: {
        signal: {
            shape: [1024, 8192],
        },
        response: {
            shape: [1024, 8192],
        }
    },
    tests: [
        {
            kind: "convo",
            signal: "signal",
            response: "response",
            repeat: 10,
        }
    ],
}
