def get_advice(consumption, production):
    """
    Very simple rule-based advisor for now.
    Later we can connect AI model + weather API.
    """
    net = consumption - production

    if production > consumption * 1.2:
        return "You have surplus solar energy. Consider turning on high-energy appliances now."
    elif net > 3:
        return "High usage detected. Try postponing non-essential appliances."
    elif net > 0:
        return "Consumption slightly exceeds production. Monitor your usage."
    elif net < -1:
        return "You are producing more than you use. Good time to store energy or sell if possible."
    else:
        return "Energy usage is balanced."
