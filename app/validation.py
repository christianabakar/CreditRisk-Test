"""Module de validation : contrat d'entrée de l'API (format ET cohérence)."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

AGE_MIN, AGE_MAX = 18, 75
AGE_MIN_ACTIVITE = 16  # âge minimal plausible de début d'activité


class DemandeCredit(BaseModel):
    """Demande de crédit. Mode strict : aucune conversion implicite de type."""

    model_config = ConfigDict(extra="forbid", strict=True)

    age: int = Field(ge=AGE_MIN, le=AGE_MAX, description="Âge du demandeur (ans)")
    revenu_mensuel: float = Field(gt=0, le=100_000_000, allow_inf_nan=False,
                                  description="Revenu mensuel (FCFA)")
    montant_credit: float = Field(gt=0, le=1_000_000_000, allow_inf_nan=False,
                                  description="Montant demandé (FCFA)")
    duree_mois: int = Field(ge=1, le=120, description="Durée du crédit (mois)")
    anciennete_activite: int = Field(ge=0, le=60,
                                     description="Ancienneté de l'activité (années)")

    @model_validator(mode="after")
    def verifier_coherence(self):
        if self.anciennete_activite > self.age - AGE_MIN_ACTIVITE:
            raise ValueError(
                f"ancienneté incohérente avec l'âge (maximum {self.age - AGE_MIN_ACTIVITE} ans)"
            )
        return self


class ReponsePrediction(BaseModel):
    probabilite_defaut: float = Field(ge=0, le=1)
    niveau_risque: str
    version_modele: str
    alertes: list[str]