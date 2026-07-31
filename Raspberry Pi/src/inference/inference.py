def run(

    audio,

    feature_extractor,

    model,

    sample_rate

):

    feature = feature_extractor(

        audio,

        sample_rate
    )

    prediction = model.predict(
        feature
    )

    return prediction