import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Preprocess:
    """
    Classe para pré-processamento de dados de corridas Uber.
    
    Responsável por:
    - Leitura e validação dos dados
    - Limpeza e transformação de colunas
    - Feature engineering para modelo de predição de cancelamentos
    """
    
    def __init__(self, file_path: Path | str) -> None:
        """
        Inicializa o pré-processador.
        
        Args:
            file_path: Caminho para o arquivo CSV com os dados das corridas
        """
        self.file_path = Path(file_path)
        self.df: Optional[pd.DataFrame] = None
        
    def read_data(self) -> pd.DataFrame:
        """
        Lê o arquivo CSV e realiza validações básicas.
        
        Returns:
            DataFrame com os dados lidos
        """
        try:
            self.df = pd.read_csv(self.file_path)
            logger.info(f"Dados carregados com sucesso: {len(self.df)} registros, {len(self.df.columns)} colunas")
            return self.df
        except FileNotFoundError:
            logger.error(f"Arquivo não encontrado: {self.file_path}")
            raise
        except Exception as e:
            logger.error(f"Erro ao ler arquivo: {e}")
            raise
    
    @staticmethod
    def drop_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Remove colunas especificadas do DataFrame.
        
        Args:
            df: DataFrame de entrada
            columns: Lista de nomes das colunas a serem removidas
            
        Returns:
            DataFrame sem as colunas especificadas
        """
        existing_cols = [col for col in columns if col in df.columns]
        if existing_cols:
            df = df.drop(columns=existing_cols)
            logger.info(f"Colunas removidas: {existing_cols}")
        else:
            logger.warning("Nenhuma das colunas especificadas foi encontrada")
        return df
    
    @staticmethod
    def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove registros duplicados.
        
        Args:
            df: DataFrame de entrada
            
        Returns:
            DataFrame sem duplicatas
        """
        initial_len = len(df)
        df = df.drop_duplicates()
        removed = initial_len - len(df)
        if removed > 0:
            logger.info(f"Removidas {removed} linhas duplicadas")
        return df
    
    @staticmethod
    def one_hot_encode(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Aplica one-hot encoding nas colunas especificadas.
        
        Args:
            df: DataFrame de entrada
            columns: Lista de colunas categóricas para codificar
            
        Returns:
            DataFrame com colunas codificadas
        """
        for col in columns:
            if col in df.columns:
                # Garantir que a coluna seja tratada como categoria
                df[col] = df[col].astype(str)
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(columns=[col])
                logger.info(f"One-hot encoding aplicado à coluna: {col} ({len(dummies.columns)} novas features)")
            else:
                logger.warning(f"Coluna '{col}' não encontrada para one-hot encoding")
        return df
    
    @staticmethod
    def timestamp_processing(df: pd.DataFrame) -> pd.DataFrame:
        """
        Processa colunas de data e hora, extraindo características temporais.
        
        Args:
            df: DataFrame de entrada
            
        Returns:
            DataFrame com colunas de tempo processadas
        """
        try:
            # Converter Date para datetime
            df['Date'] = pd.to_datetime(df['Date'], errors="coerce")
            
            # Converter Time para time object
            df['Time'] = pd.to_datetime(df['Time'], format="%H:%M:%S", errors="coerce").dt.time
            
            # Combinar Date e Time em uma única coluna datetime
            df['datetime'] = pd.to_datetime(
                df['Date'].astype(str) + ' ' + df['Time'].astype(str), 
                errors="coerce"
            )
            
            # Extrair características temporais
            df['hour'] = df['datetime'].dt.hour
            df['day'] = df['datetime'].dt.day
            df['month'] = df['datetime'].dt.month
            df['weekday'] = df['datetime'].dt.dayofweek  # Monday=0, Sunday=6
            df['is_weekend'] = df['weekday'].isin([5, 6])
            
            logger.info("Processamento de timestamp concluído")
            return df
        except Exception as e:
            logger.error(f"Erro no processamento de timestamp: {e}")
            raise
    
    @staticmethod
    def create_binary_flags(df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria flags binárias baseadas em colunas de cancelamento e incompletude.
        
        Args:
            df: DataFrame de entrada
            
        Returns:
            DataFrame com flags binárias adicionadas
        """
        flag_columns = ['Cancelled Rides by Customer', 'Cancelled Rides by Driver', 'Incomplete Rides']
        flag_names = ['is_cancelled_customer', 'is_cancelled_driver', 'is_incomplete']
        
        for col, name in zip(flag_columns, flag_names):
            if col in df.columns:
                df[name] = df[col].notnull()
                logger.info(f"Criada flag: {name}")
            else:
                logger.warning(f"Coluna '{col}' não encontrada para criação de flag")
        
        # Criar flags para valores ausentes
        missing_cols = ['Driver Ratings', 'Customer Rating', 'Booking Value', 'Payment Method']
        missing_names = ['missing_driver_rating', 'missing_customer_rating', 'missing_booking_value', 'missing_payment_method']
        
        for col, name in zip(missing_cols, missing_names):
            if col in df.columns:
                df[name] = df[col].isnull()
                logger.info(f"Criada flag: {name}")
        
        return df
    
    @staticmethod
    def fill_missing_values(df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
        """
        Preenche faltantes ausentes em colunas numéricas.
        
        Args:
            df: DataFrame de entrada
            strategy: Estratégia de preenchimento ('median', 'mean', 'mode', 'constant')
            
        Returns:
            DataFrame com valores ausentes preenchidos
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].isnull().any():
                if strategy == 'median':
                    fill_value = df[col].median()
                elif strategy == 'mean':
                    fill_value = df[col].mean()
                elif strategy == 'mode':
                    fill_value = df[col].mode()[0] if not df[col].mode().empty else 0
                else:
                    fill_value = 0
                
                df[col].fillna(fill_value, inplace=True)
                logger.info(f"Preenchidos valores ausentes em '{col}' com {strategy}: {fill_value:.2f}")
        
        # Preencher colunas de texto com 'Unknown'
        string_cols = df.select_dtypes(include=['object']).columns
        for col in string_cols:
            df[col].fillna('Unknown', inplace=True)
        
        return df
    
    @staticmethod
    def create_customer_frequency(df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria feature de frequência de reservas por cliente.
        
        Args:
            df: DataFrame de entrada
            
        Returns:
            DataFrame com coluna 'customer_total_bookings' adicionada
        """
        if 'Customer ID' in df.columns:
            cust_counts = df['Customer ID'].value_counts().to_dict()
            df['customer_total_bookings'] = df['Customer ID'].map(cust_counts)
            logger.info("Criada feature: customer_total_bookings")
        else:
            logger.warning("Coluna 'Customer ID' não encontrada para criação de frequência")
        return df
    
    @staticmethod
    def create_target(df: pd.DataFrame) -> pd.DataFrame:
        df['target_customer_cancelled'] = (
        df['Cancelled Rides by Customer']
        .notnull()
        .astype(int)
    )
        return df
    
    @staticmethod
    def encode_top_locations(df: pd.DataFrame, 
                            location_col: str, 
                            n_top: int = 10) -> pd.DataFrame:
        """
        Codifica as top N localizações, agrupando as demais como 'Other'.
        
        Args:
            df: DataFrame de entrada
            location_col: Nome da coluna de localização
            n_top: Número de localizações principais a manter
            
        Returns:
            DataFrame com coluna de localização codificada
        """
        if location_col in df.columns:
            top_locations = df[location_col].value_counts().nlargest(n_top).index
            encoded_col = f'{location_col.lower().replace(" ", "_")}_encoded'
            df[encoded_col] = df[location_col].apply(
                lambda x: x if x in top_locations else 'Other'
            )
            logger.info(f"Localizações em '{location_col}' codificadas: {len(top_locations)} principais mantidas")
            return df
        else:
            logger.warning(f"Coluna '{location_col}' não encontrada")
            return df
    
    @staticmethod
    def separate_features_target(df: pd.DataFrame, 
                                 target_col: str = 'target_customer_cancelled',
                                 leaky_features: Optional[List[str]] = None) -> tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Separa features e target, removendo colunas com vazamento de dados.
        
        Args:
            df: DataFrame de entrada
            target_col: Nome da coluna target
            leaky_features: Lista de colunas que devem ser removidas por vazamento de dados
            
        Returns:
            Tupla (X, y, feature_names)
        """
        if leaky_features is None:
            leaky_features = [
                'Avg VTAT', 'Ride Distance', 'Booking Value', 
                'Customer Rating', 'Driver Ratings', 'Booking Status', 
                'Payment Method', 'missing_booking_value', 'missing_payment_method',
                'missing_driver_rating', 'missing_customer_rating',
                'is_cancelled_customer', 'is_cancelled_driver', 'is_incomplete'
            ]
        
        # Separar target
        if target_col not in df.columns:
            raise ValueError(f"Coluna target '{target_col}' não encontrada")
        
        y = df[target_col]
        
        # Remover target e colunas com vazamento
        features = [col for col in df.columns if col != target_col]
        features = [col for col in features if col not in leaky_features]
        
        X = df[features]
        
        logger.info(f"Features para modelagem: {len(features)}")
        logger.info(f"Target: {target_col} (distribuição: {y.value_counts(normalize=True).to_dict()})")
        
        return X, y, features
    
    def run_pipeline(self, 
                    drop_original_cols: Optional[List[str]] = None,
                    categorical_cols: Optional[List[str]] = None,
                    target_col: str = 'target_customer_cancelled') -> tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Executa o pipeline completo de pré-processamento.
        
        Args:
            drop_original_cols: Lista de colunas originais para remover
            categorical_cols: Lista de colunas categóricas para one-hot encoding
            target_col: Nome da coluna target
            
        Returns:
            Tupla (X, y, feature_names) prontas para modelagem
        """
        # Configurações padrão
        if drop_original_cols is None:
            drop_original_cols = [
                'Booking ID', 'Customer ID', 'Pickup Location', 'Drop Location',
                'Cancelled Rides by Customer', 'Reason for cancelling by Customer',
                'Cancelled Rides by Driver', 'Driver Cancellation Reason',
                'Incomplete Rides', 'Incomplete Rides Reason', 'Date', 'Time', 'datetime'
            ]
        
        if categorical_cols is None:
            categorical_cols = ['Vehicle Type', 'Payment Method', 'Booking Status']
        
        # Executar pipeline
        self.read_data()
        self.df = self.drop_duplicates(self.df)
        self.df = self.timestamp_processing(self.df)
        self.df = self.create_target(self.df)
        self.df = self.create_binary_flags(self.df)
        self.df = self.fill_missing_values(self.df, strategy='median')
        self.df = self.create_customer_frequency(self.df)
        
        # Codificar localizações
        self.df = self.encode_top_locations(self.df, 'Pickup Location', n_top=10)
        self.df = self.encode_top_locations(self.df, 'Drop Location', n_top=10)
        
        # One-hot encoding para colunas categóricas
        self.df = self.one_hot_encode(self.df, categorical_cols)
        
        # One-hot encoding para localizações codificadas
        if 'pickup_location_encoded' in self.df.columns:
            self.df = self.one_hot_encode(self.df, ['pickup_location_encoded'])
        if 'drop_location_encoded' in self.df.columns:
            self.df = self.one_hot_encode(self.df, ['drop_location_encoded'])
        
        # Remover colunas originais desnecessárias
        self.df = self.drop_columns(self.df, drop_original_cols)
        
        # Separar features e target
        X, y, features = self.separate_features_target(self.df, target_col)
        
        logger.info("Pipeline de pré-processamento concluído com sucesso")
        return X, y, features


if __name__ == "__main__":
    # Caminho para os dados
    data_path = Path("/home/anderson/Documents/uber_dataset/data/ncr_ride_bookings.csv")
    
    # Inicializar e executar pipeline
    preprocessor = Preprocess(data_path)
    X, y, features = preprocessor.run_pipeline()
    
    print(f"\nResultados:")
    print(f"Features: {len(features)}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Distribuição do target:\n{y.value_counts(normalize=True)}")