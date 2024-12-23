import torch
from sklearn.metrics import precision_score, recall_score, f1_score

class NetowrkTrainer:
    
    @staticmethod
    def train_loop(dataloader, model, criterion, optimizer, epochs):
        for epoch in range(epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            correct = 0
            total = 0
            for i, data in enumerate(dataloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                
                running_loss += loss.item()  # Aggiungi la perdita alla somma totale
                # Calcola le previsioni
                _, predicted = torch.max(outputs, 1)  # Trova l'indice della classe con probabilità massima
                total += labels.size(0)  # Aggiungi il numero di immagini processate
                correct += (predicted == labels).sum().item()  # Conta le predizioni corrette

            # Stampa il progresso dopo ogni epoca
            print(f'Epoca [{epoch+1}/{epochs}], \
                    Loss: {running_loss/len(dataloader):.4f}, \
                    Accuracy: {100 * correct / total:.2f}%')

        print('Finished Training')

    @staticmethod
    def test_loop(testloader, model):
        model.eval()
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        
        with torch.no_grad():
            for inputs, labels in testloader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')
    
        print(f"Accuratezza sul test set: {100 * correct / total:.2f}%")
        print(f"Precisione: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
